"""Tests for rioxarray interoperability."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio
from affine import Affine


def test_rioxarray_reads_lazycogs_spatial_metadata(opened_dataarray):
    """rioxarray reads 2D slice metadata without repair."""
    pytest.importorskip("rioxarray")

    band = opened_dataarray.isel(time=0, band=0, drop=True)

    assert band.rio.x_dim == "x"
    assert band.rio.y_dim == "y"
    assert band.rio.crs.to_epsg() == 32632
    assert band.rio.transform() == Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
    assert band.rio.nodata == 0
    assert band.rio.encoded_nodata is None
    assert "_FillValue" not in band.encoding


def _assert_raster_metadata(path, *, count):
    with rasterio.open(path) as dataset:
        assert dataset.count == count
        assert dataset.width == 10
        assert dataset.height == 10
        assert dataset.crs.to_epsg() == 32632
        assert dataset.transform == Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
        assert dataset.nodata == 0


def test_rioxarray_writes_2d_image_slice_without_metadata_repair(
    opened_dataarray,
    tmp_path,
):
    """A 2D image slice can be exported through rioxarray as a one-band raster."""
    pytest.importorskip("rioxarray")

    band = opened_dataarray.isel(time=0, band=0, drop=True)
    exportable_band = band.copy(data=np.ones(band.shape, dtype=band.dtype))
    output_path = tmp_path / "band.tif"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning)
        exportable_band.rio.to_raster(output_path)

    _assert_raster_metadata(output_path, count=1)


def test_rioxarray_writes_3d_band_stack_without_metadata_repair(
    opened_dataarray,
    tmp_path,
):
    """A 3D band stack can be exported through rioxarray as a multiband raster."""
    pytest.importorskip("rioxarray")

    scene = opened_dataarray.isel(time=0, drop=True)
    exportable_scene = scene.copy(data=np.ones(scene.shape, dtype=scene.dtype))
    output_path = tmp_path / "scene.tif"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning)
        exportable_scene.rio.to_raster(output_path)

    _assert_raster_metadata(output_path, count=scene.sizes["band"])
