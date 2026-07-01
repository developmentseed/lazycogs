"""Tests for open_cog: single-COG reads at native resolution."""

from __future__ import annotations

import numpy as np
import pytest
from obstore.store import LocalStore
from pyproj import CRS

import lazycogs


@pytest.fixture
def native_da(synthetic_cog):
    """Open the synthetic COG at native resolution via a local obstore store."""
    store = LocalStore()
    return lazycogs.open_cog(synthetic_cog.as_uri(), store=store)


def test_native_shape_matches_source(native_da):
    """No reprojection: output keeps the COG's 2048 x 2048 native shape."""
    assert native_da.dims == ("band", "y", "x")
    assert native_da.sizes == {"band": 1, "y": 2048, "x": 2048}


def test_native_crs_preserved(native_da):
    """The native UTM 32N CRS is preserved, not reprojected."""
    crs = CRS.from_wkt(native_da["spatial_ref"].attrs["crs_wkt"])
    assert crs.to_epsg() == 32632


def test_native_resolution_preserved(native_da):
    """Native 10 m pixel size is preserved on both axes."""
    x = native_da["x"].to_numpy()
    y = native_da["y"].to_numpy()
    assert np.isclose(abs(x[1] - x[0]), 10.0)
    assert np.isclose(abs(y[1] - y[0]), 10.0)


def test_nodata_advertised_as_fillvalue(native_da):
    """Source nodata of 0 is surfaced as _FillValue, grid_mapping is set."""
    assert native_da.attrs["_FillValue"] == 0
    assert native_da.attrs["grid_mapping"] == "spatial_ref"


def test_values_loaded(native_da):
    """Pixel values are read eagerly and finite."""
    data = native_da.to_numpy()
    assert data.shape == (1, 2048, 2048)
    assert data.max() > 0
