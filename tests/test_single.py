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


def _item(**assets) -> dict:
    """Build a minimal STAC item dict from `key=cog_path` pairs."""
    return {
        "assets": {
            key: {
                "href": path.as_uri(),
                "roles": ["data"],
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            }
            for key, path in assets.items()
        },
    }


@pytest.fixture
def two_band_item(synthetic_cog, synthetic_cog_b):
    """A STAC item with two same-grid single-band assets."""
    return _item(b04=synthetic_cog, b08=synthetic_cog_b)


def test_open_item_stacks_bands_by_asset_key(two_band_item):
    """Selected assets stack into (band, y, x) labelled by asset key."""
    store = LocalStore()
    da = lazycogs.open_item(two_band_item, bands=["b04", "b08"], store=store)

    assert da.dims == ("band", "y", "x")
    assert da.sizes == {"band": 2, "y": 2048, "x": 2048}
    assert list(da["band"].to_numpy()) == ["b04", "b08"]
    # Distinct source COGs → distinct band data.
    assert not np.array_equal(da.isel(band=0).to_numpy(), da.isel(band=1).to_numpy())


def test_open_item_defaults_to_preferred_data_assets(two_band_item):
    """Omitting bands uses the item's preferred data assets."""
    store = LocalStore()
    da = lazycogs.open_item(two_band_item, store=store)
    assert set(da["band"].to_numpy()) == {"b04", "b08"}


def test_open_item_preserves_native_crs_and_resolution(two_band_item):
    """open_item keeps the native grid; no reprojection."""
    store = LocalStore()
    da = lazycogs.open_item(two_band_item, bands=["b04", "b08"], store=store)

    crs = CRS.from_wkt(da["spatial_ref"].attrs["crs_wkt"])
    assert crs.to_epsg() == 32632
    x = da["x"].to_numpy()
    assert np.isclose(abs(x[1] - x[0]), 10.0)


def test_open_item_surfaces_shared_nodata(two_band_item):
    """A nodata value shared by all bands is advertised as _FillValue."""
    store = LocalStore()
    da = lazycogs.open_item(two_band_item, bands=["b04", "b08"], store=store)
    assert da.attrs["_FillValue"] == 0
    assert da.attrs["grid_mapping"] == "spatial_ref"


def test_open_item_rejects_grid_mismatch(synthetic_cog, synthetic_cog_offgrid):
    """Assets on different native grids raise a ValueError."""
    item = _item(b04=synthetic_cog, off=synthetic_cog_offgrid)
    store = LocalStore()
    with pytest.raises(ValueError, match="native grid"):
        lazycogs.open_item(item, bands=["b04", "off"], store=store)


def test_open_item_rejects_multiband_asset(synthetic_cog, synthetic_cog_multiband):
    """A multi-band asset raises and points to open_cog."""
    item = _item(b04=synthetic_cog, mb=synthetic_cog_multiband)
    store = LocalStore()
    with pytest.raises(ValueError, match="single-band"):
        lazycogs.open_item(item, bands=["b04", "mb"], store=store)


def test_open_item_rejects_unknown_band(two_band_item):
    """Requesting a band absent from the item raises a ValueError."""
    store = LocalStore()
    with pytest.raises(ValueError, match="not present"):
        lazycogs.open_item(two_band_item, bands=["b04", "missing"], store=store)


def test_open_item_rejects_item_without_assets():
    """An item with no assets raises a ValueError."""
    with pytest.raises(ValueError, match="no assets"):
        lazycogs.open_item({"assets": {}})
