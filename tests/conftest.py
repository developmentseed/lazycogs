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


@pytest.fixture(scope="session")
def synthetic_cog(tmp_path_factory) -> Path:
    """Write a small synthetic COG with four overview levels to a temp file.

    Properties:
    - Native resolution: 10 m, 320 x 320 pixels
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
    cog_path = tmp_path_factory.mktemp("cog") / "synthetic.tif"
    native_res = 10.0
    size = 2048
    minx, maxy = 500_000.0, 5_600_000.0
    transform = Affine(native_res, 0.0, minx, 0.0, -native_res, maxy)
    crs_wkt = CRS.from_epsg(32632).to_wkt()

    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    data = ((cols + rows * size) % 65535 + 1).astype(np.uint16)

    # Step 1: write to a temporary stripped GeoTIFF and build overviews.
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with rasterio.open(
        tmp_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint16",
        crs=crs_wkt,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data[np.newaxis])

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
