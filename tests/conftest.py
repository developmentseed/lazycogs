"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import rasterio.enums
import rasterio.shutil
import rustac
from affine import Affine
from obstore.store import MemoryStore
from pyproj import CRS

import lazycogs
from lazycogs import _store


def clear_store_cache_for_tests() -> None:
    """Clear the shared store cache between tests that exercise resolve()."""
    with _store._STORE_CACHE_LOCK:
        _store._STORE_CACHE.clear()


def _fake_open_item() -> dict:
    return {
        "id": "test-item",
        "stac_extensions": [],
        "properties": {"datetime": "2023-01-15T10:00:00Z"},
        "assets": {
            "B04": {
                "href": "s3://bucket/B04.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"],
            },
        },
    }


def _items_to_arrow(items: list[dict]) -> rustac.DuckdbClient:
    if not items:
        return None
    full_items = []
    for i, item in enumerate(items):
        props = dict(item.get("properties", {}))
        full_items.append(
            {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": f"fake-{i}",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "bbox": [-0.1, -0.1, 0.1, 0.1],
                "properties": props,
                "links": [],
                "assets": {},
            },
        )
    return rustac.to_arrow(full_items)


@pytest.fixture
def clear_store_cache() -> Iterator[None]:
    """Reset the shared store cache around a test."""
    clear_store_cache_for_tests()
    yield
    clear_store_cache_for_tests()


@pytest.fixture
def opened_dataarray(tmp_path):
    """Return a small DataArray from open() with DuckDB calls patched."""
    parquet = tmp_path / "items.parquet"
    parquet.write_bytes(b"")

    store = MemoryStore()
    store.put("B04.tif", b"dummy")

    table = _items_to_arrow([{"properties": {"datetime": "2023-01-15T10:00:00Z"}}])

    class _FakeGeoTIFF:
        dtype = np.dtype("uint16")
        nodata = 0

    async def fake_open(path: str, *, store):
        return _FakeGeoTIFF()

    with (
        patch("rustac.DuckdbClient.search", return_value=[_fake_open_item()]),
        patch("rustac.DuckdbClient.search_to_arrow", return_value=table),
        patch("lazycogs._core.GeoTIFF.open", side_effect=fake_open),
    ):
        return lazycogs.open(
            str(parquet),
            bbox=(0.0, 0.0, 100.0, 100.0),
            crs="EPSG:32632",
            resolution=10.0,
            store=store,
            path_from_href=lambda href: href.split("/", 3)[-1],
        )


def _write_synthetic_cog(
    cog_path: Path,
    *,
    size: int = 2048,
    native_res: float = 10.0,
    minx: float = 500_000.0,
    maxy: float = 5_600_000.0,
    epsg: int = 32632,
    count: int = 1,
    nodata: float | None = 0,
    seed: int = 0,
) -> Path:
    """Write a tiled synthetic COG with four overview levels.

    Pixel values are unique per pixel (``col + row * size`` plus a per-band and
    per-``seed`` offset) so tests can tell which source pixel and band was
    sampled. The two-step recipe keeps both the full-resolution IFD and every
    overview IFD tiled, which async_geotiff requires.

    Args:
        cog_path: Destination path for the COG.
        size: Width and height in pixels.
        native_res: Pixel size in CRS units.
        minx: Left edge (origin easting).
        maxy: Top edge (origin northing).
        epsg: CRS EPSG code.
        count: Number of bands.
        nodata: Nodata value, or ``None`` for no nodata.
        seed: Offset added to pixel values so distinct COGs differ.

    Returns:
        ``cog_path``.
    """
    transform = Affine(native_res, 0.0, minx, 0.0, -native_res, maxy)
    crs_wkt = CRS.from_epsg(epsg).to_wkt()

    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    linear = cols + rows * size + seed
    # count=1, seed=0 reproduces the original single-band fixture exactly:
    # ((cols + rows * size) % 65535 + 1).
    data = np.stack(
        [
            ((linear + band * 100) % 65535 + 1).astype(np.uint16)
            for band in range(count)
        ],
    )

    # Step 1: write to a temporary stripped GeoTIFF and build overviews.
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with rasterio.open(
        tmp_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype="uint16",
        crs=crs_wkt,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)

    with rasterio.open(tmp_path, "r+") as dst:
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")

    # Step 2: copy to a tiled COG so async_geotiff can read all IFDs.
    rasterio.shutil.copy(
        str(tmp_path),
        str(cog_path),
        driver="GTiff",
        copy_src_overviews=True,
        tiled=True,
        blockxsize=64,
        blockysize=64,
    )
    tmp_path.unlink()

    return cog_path


@pytest.fixture(scope="session")
def synthetic_cog(tmp_path_factory) -> Path:
    """Write a small synthetic COG with four overview levels to a temp file.

    Properties:
    - Native resolution: 10 m, 2048 x 2048 pixels
    - CRS: UTM zone 32N (EPSG:32632)
    - Origin: 500 000 E, 5 600 000 N
    - Overview shrink factors: [2, 4, 8, 16] → resolutions 20, 40, 80, 160 m
    - Pixel values: unique uint16 per pixel (col + row * width), so every
      sampling position returns a deterministic, distinct value that lets
      tests distinguish which source pixel was sampled.
    - Nodata: 0 (pixels shifted by 1 to avoid accidental nodata)

    The file is written using the standard two-step COG recipe so that both
    the full-resolution IFD and all overview IFDs are tiled (required by
    async_geotiff).
    """
    return _write_synthetic_cog(tmp_path_factory.mktemp("cog") / "synthetic.tif")


@pytest.fixture(scope="session")
def synthetic_cog_b(tmp_path_factory) -> Path:
    """A second single-band COG on the same grid as ``synthetic_cog``.

    Different pixel values (``seed``) so a stacked ``open_item`` result carries
    distinct data per band.
    """
    return _write_synthetic_cog(
        tmp_path_factory.mktemp("cog_b") / "synthetic_b.tif",
        seed=1000,
    )


@pytest.fixture(scope="session")
def synthetic_cog_offgrid(tmp_path_factory) -> Path:
    """A single-band COG on a different grid (20 m, shifted origin).

    Used to check that ``open_item`` rejects assets that do not share one
    native grid.
    """
    return _write_synthetic_cog(
        tmp_path_factory.mktemp("cog_offgrid") / "synthetic_offgrid.tif",
        native_res=20.0,
        minx=600_000.0,
    )


@pytest.fixture(scope="session")
def synthetic_cog_multiband(tmp_path_factory) -> Path:
    """A two-band COG on the ``synthetic_cog`` grid.

    Used to check that ``open_item`` rejects multi-band assets.
    """
    return _write_synthetic_cog(
        tmp_path_factory.mktemp("cog_mb") / "synthetic_mb.tif",
        count=2,
    )
