"""Offline regression tests backed by the local benchmark dataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from shutil import copy2
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest
import rasterio
import rustac
from pyproj import Transformer
from rasterio.windows import Window

import lazycogs
from lazycogs import LowestMethod

from .conftest import (
    BENCHMARK_BBOX,
    BENCHMARK_CRS,
    BENCHMARK_MULTIBAND,
    BENCHMARK_NATIVE_CRS,
    BENCHMARK_SINGLE_BAND,
)


def _path_from_href(href: str) -> Path:
    """Return the local filesystem path for a ``file://`` benchmark asset HREF."""
    return Path(urlparse(href).path)


def _write_asset_variant(
    source_href: str,
    dest: Path,
    *,
    nodata: float | None,
    pixel_updates: list[tuple[int, int, int]] | None = None,
    dtype: str | None = None,
) -> str:
    """Copy one benchmark GeoTIFF and apply metadata/data tweaks in place."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dtype is None:
        copy2(_path_from_href(source_href), dest)
        with rasterio.open(dest, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as dataset:
            dataset.nodata = nodata
            if pixel_updates:
                for row, col, value in pixel_updates:
                    patch = np.array([[value]], dtype=dataset.dtypes[0])
                    dataset.write(patch, 1, window=Window(col, row, 1, 1))
        return dest.as_uri()

    with rasterio.open(_path_from_href(source_href)) as source:
        profile = source.profile.copy()
        profile.update(dtype=dtype, tiled=True)
        data = source.read().astype(dtype)

    with rasterio.open(dest, "w", **profile) as dataset:
        dataset.write(data)
        dataset.nodata = nodata
        if pixel_updates:
            for row, col, value in pixel_updates:
                patch = np.array([[value]], dtype=dtype)
                dataset.write(patch, 1, window=Window(col, row, 1, 1))

    return dest.as_uri()


def _write_items_parquet(path: Path, items: list[dict[str, Any]]) -> str:
    """Write a temporary local parquet file for a derived benchmark scenario."""
    rustac.write_sync(str(path), items)
    return str(path)


def _pixel_bbox(
    source_href: str,
    *,
    row: int,
    col: int,
) -> tuple[float, float, float, float]:
    """Return the bounding box of one source pixel in the asset's native CRS."""
    with rasterio.open(_path_from_href(source_href)) as dataset:
        left, top = dataset.transform * (col, row)
        right, bottom = dataset.transform * (col + 1, row + 1)
    return (left, bottom, right, top)


def _item_center_pixel(item: dict[str, Any], band: str) -> tuple[int, int]:
    """Project the STAC item's bbox center into the asset grid and return row/col."""
    source_href = item["assets"][band]["href"]
    with rasterio.open(_path_from_href(source_href)) as dataset:
        transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
        minx, miny, maxx, maxy = item["bbox"]
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        x_native, y_native = transformer.transform(center_x, center_y)
        return dataset.index(x_native, y_native)


def test_open_infers_uint16_and_coherent_nodata_from_benchmark_data(
    benchmark_parquet: str,
) -> None:
    """The local benchmark parquet exercises the inferred dtype/nodata contract."""
    da = lazycogs.open(
        benchmark_parquet,
        bbox=BENCHMARK_BBOX,
        crs=BENCHMARK_CRS,
        resolution=60.0,
        bands=BENCHMARK_MULTIBAND,
    )

    assert da.dtype == np.dtype("uint16")
    assert da.attrs["nodata"] == 0
    assert da.attrs["_FillValue"] == 0
    assert da.attrs["missing_value"] == 0


def test_open_explicit_overrides_win_over_benchmark_inference(
    benchmark_parquet: str,
) -> None:
    """Caller-supplied dtype and nodata stay authoritative on local data."""
    da = lazycogs.open(
        benchmark_parquet,
        bbox=BENCHMARK_BBOX,
        crs=BENCHMARK_CRS,
        resolution=60.0,
        bands=BENCHMARK_SINGLE_BAND,
        dtype="float32",
        nodata=-9999,
    )

    assert da.dtype == np.dtype("float32")
    assert da.attrs["nodata"] == -9999
    assert da.attrs["_FillValue"] == -9999
    assert da.attrs["missing_value"] == -9999


def test_open_rejects_conflicting_sampled_nodata_on_local_benchmark_copy(
    tmp_path: Path,
    benchmark_items: list[dict[str, Any]],
) -> None:
    """A derived offline parquet with band conflicts fails fast at ``open()``."""
    item = deepcopy(benchmark_items[0])
    item["assets"]["nir08"]["href"] = _write_asset_variant(
        item["assets"]["nir08"]["href"],
        tmp_path / "conflict" / "nir08-conflict.tif",
        nodata=-9999,
    )
    parquet_path = _write_items_parquet(tmp_path / "conflict.parquet", [item])

    with pytest.raises(ValueError, match="Conflicting sampled nodata values"):
        lazycogs.open(
            parquet_path,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            bands=BENCHMARK_MULTIBAND,
        )


def test_open_accepts_conflicting_sampled_nodata_with_explicit_override(
    tmp_path: Path,
    benchmark_items: list[dict[str, Any]],
) -> None:
    """The same offline conflict opens successfully when ``nodata=`` is explicit."""
    item = deepcopy(benchmark_items[0])
    item["assets"]["nir08"]["href"] = _write_asset_variant(
        item["assets"]["nir08"]["href"],
        tmp_path / "override" / "nir08-conflict.tif",
        nodata=-9999,
    )
    parquet_path = _write_items_parquet(tmp_path / "override.parquet", [item])

    da = lazycogs.open(
        parquet_path,
        bbox=BENCHMARK_BBOX,
        crs=BENCHMARK_CRS,
        resolution=60.0,
        bands=BENCHMARK_MULTIBAND,
        nodata=-9999,
    )

    assert da.attrs["nodata"] == -9999
    assert da.attrs["_FillValue"] == -9999
    assert da.attrs["missing_value"] == -9999


def test_compute_raises_on_later_incompatible_auto_inferred_dtype(
    tmp_path: Path,
    benchmark_items: list[dict[str, Any]],
) -> None:
    """Compute fails loudly when a later item violates inferred dtype."""
    first_item = deepcopy(benchmark_items[0])
    row, col = _item_center_pixel(first_item, "red")
    pixel_bbox = _pixel_bbox(first_item["assets"]["red"]["href"], row=row, col=col)
    first_item["assets"]["red"]["href"] = _write_asset_variant(
        first_item["assets"]["red"]["href"],
        tmp_path / "dtype-conflict" / "red-masked-first.tif",
        nodata=0,
        pixel_updates=[(row, col, 0)],
    )

    second_item = deepcopy(first_item)
    second_item["id"] = f"{first_item['id']}-float32"
    second_item["assets"]["red"]["href"] = _write_asset_variant(
        benchmark_items[0]["assets"]["red"]["href"],
        tmp_path / "dtype-conflict" / "red-float32.tif",
        nodata=0,
        dtype="float32",
    )

    parquet_path = _write_items_parquet(
        tmp_path / "dtype-conflict.parquet",
        [first_item, second_item],
    )

    da = lazycogs.open(
        parquet_path,
        bbox=pixel_bbox,
        crs=BENCHMARK_NATIVE_CRS,
        resolution=10.0,
        bands=BENCHMARK_SINGLE_BAND,
    )

    with pytest.raises(ValueError, match="Auto-inferred output dtype"):
        da.compute()


def test_compute_accepts_mixed_dtypes_with_explicit_dtype(
    tmp_path: Path,
    benchmark_items: list[dict[str, Any]],
) -> None:
    """Explicit output dtype remains authoritative for mixed source dtypes."""
    first_item = deepcopy(benchmark_items[0])
    row, col = _item_center_pixel(first_item, "red")
    pixel_bbox = _pixel_bbox(first_item["assets"]["red"]["href"], row=row, col=col)
    first_item["assets"]["red"]["href"] = _write_asset_variant(
        first_item["assets"]["red"]["href"],
        tmp_path / "dtype-explicit" / "red-masked-first.tif",
        nodata=0,
        pixel_updates=[(row, col, 0)],
    )

    second_item = deepcopy(first_item)
    second_item["id"] = f"{first_item['id']}-float32"
    second_item["assets"]["red"]["href"] = _write_asset_variant(
        benchmark_items[0]["assets"]["red"]["href"],
        tmp_path / "dtype-explicit" / "red-float32.tif",
        nodata=0,
        dtype="float32",
        pixel_updates=[(row, col, 5)],
    )

    parquet_path = _write_items_parquet(
        tmp_path / "dtype-explicit.parquet",
        [first_item, second_item],
    )

    da = lazycogs.open(
        parquet_path,
        bbox=pixel_bbox,
        crs=BENCHMARK_NATIVE_CRS,
        resolution=10.0,
        bands=BENCHMARK_SINGLE_BAND,
        dtype="float32",
    )

    data = da.compute()

    assert data.dtype == np.dtype("float32")
    assert data.item() == pytest.approx(5.0)


def test_compute_raises_on_later_conflicting_auto_inferred_nodata(
    tmp_path: Path,
    benchmark_items: list[dict[str, Any]],
) -> None:
    """Compute fails loudly when a later item violates inferred nodata."""
    first_item = deepcopy(benchmark_items[0])
    row, col = _item_center_pixel(first_item, "red")
    pixel_bbox = _pixel_bbox(first_item["assets"]["red"]["href"], row=row, col=col)
    first_item["assets"]["red"]["href"] = _write_asset_variant(
        first_item["assets"]["red"]["href"],
        tmp_path / "nodata-conflict" / "red-masked-first.tif",
        nodata=0,
        pixel_updates=[(row, col, 0)],
    )

    second_item = deepcopy(first_item)
    second_item["id"] = f"{first_item['id']}-nodata-conflict"
    second_item["assets"]["red"]["href"] = _write_asset_variant(
        benchmark_items[0]["assets"]["red"]["href"],
        tmp_path / "nodata-conflict" / "red-nodata-conflict.tif",
        nodata=-9999,
        pixel_updates=[(row, col, 7)],
    )

    parquet_path = _write_items_parquet(
        tmp_path / "nodata-conflict.parquet",
        [first_item, second_item],
    )

    da = lazycogs.open(
        parquet_path,
        bbox=pixel_bbox,
        crs=BENCHMARK_NATIVE_CRS,
        resolution=10.0,
        bands=BENCHMARK_SINGLE_BAND,
    )

    with pytest.raises(ValueError, match="Auto-inferred output nodata"):
        da.compute()


def test_compute_accepts_conflicting_nodata_with_explicit_override(
    tmp_path: Path,
    benchmark_items: list[dict[str, Any]],
) -> None:
    """Explicit output nodata remains authoritative for mixed source nodata."""
    first_item = deepcopy(benchmark_items[0])
    row, col = _item_center_pixel(first_item, "red")
    pixel_bbox = _pixel_bbox(first_item["assets"]["red"]["href"], row=row, col=col)
    first_item["assets"]["red"]["href"] = _write_asset_variant(
        first_item["assets"]["red"]["href"],
        tmp_path / "nodata-explicit" / "red-masked-first.tif",
        nodata=0,
        pixel_updates=[(row, col, 0)],
    )

    second_item = deepcopy(first_item)
    second_item["id"] = f"{first_item['id']}-nodata-conflict"
    second_item["assets"]["red"]["href"] = _write_asset_variant(
        benchmark_items[0]["assets"]["red"]["href"],
        tmp_path / "nodata-explicit" / "red-nodata-conflict.tif",
        nodata=-9999,
        pixel_updates=[(row, col, 7)],
    )

    da = lazycogs.open(
        _write_items_parquet(
            tmp_path / "nodata-explicit.parquet",
            [first_item, second_item],
        ),
        bbox=pixel_bbox,
        crs=BENCHMARK_NATIVE_CRS,
        resolution=10.0,
        bands=BENCHMARK_SINGLE_BAND,
        nodata=0,
    )

    data = da.compute()

    assert data.shape == (1, 1, 1, 1)
    assert data.item() == 7
    assert da.attrs["nodata"] == 0


def test_chunk_reads_still_mask_per_cog_nodata_with_explicit_override(
    tmp_path: Path,
    benchmark_items: list[dict[str, Any]],
) -> None:
    """Chunk reads still honor masked source pixels with explicit nodata."""
    base_item = deepcopy(benchmark_items[0])
    first_href = base_item["assets"]["red"]["href"]
    test_row, test_col = _item_center_pixel(base_item, "red")
    pixel_bbox = _pixel_bbox(first_href, row=test_row, col=test_col)

    item_without_output_nodata = deepcopy(base_item)
    item_without_output_nodata["id"] = f"{base_item['id']}-no-output-nodata"
    item_without_output_nodata["assets"] = {
        "red": {
            **base_item["assets"]["red"],
            "href": _write_asset_variant(
                first_href,
                tmp_path / "masking" / "red-no-output-nodata.tif",
                nodata=None,
                pixel_updates=[(test_row, test_col, 10)],
            ),
        },
    }

    masked_source_item = deepcopy(base_item)
    masked_source_item["id"] = f"{base_item['id']}-masked-source"
    masked_source_item["assets"] = {
        "red": {
            **base_item["assets"]["red"],
            "href": _write_asset_variant(
                first_href,
                tmp_path / "masking" / "red-source-nodata.tif",
                nodata=0,
                pixel_updates=[(test_row, test_col, 0)],
            ),
        },
    }

    parquet_path = _write_items_parquet(
        tmp_path / "masking.parquet",
        [item_without_output_nodata, masked_source_item],
    )

    da = lazycogs.open(
        parquet_path,
        bbox=pixel_bbox,
        crs=BENCHMARK_NATIVE_CRS,
        resolution=10.0,
        bands=BENCHMARK_SINGLE_BAND,
        mosaic_method=LowestMethod,
        nodata=0,
    )
    value = da.compute().item()

    assert da.attrs["nodata"] == 0
    assert da.attrs["_FillValue"] == 0
    assert da.attrs["missing_value"] == 0
    assert value == 10
