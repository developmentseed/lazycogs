"""Tests for open_cog: single-COG reads at native resolution."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import xarray as xr
from async_geotiff import GeoTIFF, Window
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
    """Materialized pixel values are finite."""
    data = native_da.to_numpy()
    assert data.shape == (1, 2048, 2048)
    assert data.max() > 0


def _item(**assets) -> dict:
    """Build a minimal STAC item dict from ``key=cog_path`` pairs."""
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


@pytest.fixture
def anyio_backend():
    """Run async tests on asyncio, the backend used by async-geotiff."""
    return "asyncio"


@pytest.fixture
def pixel_reads(monkeypatch):
    """Record full-resolution reads while retaining real local COG I/O."""
    reads = []
    original = GeoTIFF.read

    async def record(geotiff, *, window=None):
        reads.append(window)
        return await original(geotiff, window=window)

    monkeypatch.setattr(GeoTIFF, "read", record)
    return reads


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_cog_selection_reads_only_native_window(
    synthetic_cog_multiband,
    pixel_reads,
    asynchronous,
    anyio_backend,
):
    """Opening and selecting are header-only; materialization reads a window."""
    href = synthetic_cog_multiband.as_uri()
    if asynchronous:
        da = await lazycogs.open_cog_async(href, store=LocalStore())
    else:
        da = lazycogs.open_cog(href, store=LocalStore())
    assert pixel_reads == []
    assert da.dtype == np.dtype("uint16")
    assert da.attrs["_FillValue"] == 0
    selected = da.isel(band=[1, 0], y=slice(17, 21), x=slice(23, 28))
    assert pixel_reads == []
    loaded = await selected.load_async() if asynchronous else selected.load()
    assert pixel_reads == [Window(col_off=23, row_off=17, width=5, height=4)]
    rows = np.arange(17, 21)[:, None]
    cols = np.arange(23, 28)[None, :]
    expected = np.stack([((cols + rows * 2048 + b * 100) % 65535 + 1) for b in [1, 0]])
    np.testing.assert_array_equal(loaded.to_numpy(), expected)
    assert list(loaded.band.values) == [2, 1]
    assert loaded.dtype == np.dtype("uint16")
    assert CRS.from_wkt(loaded.spatial_ref.attrs["crs_wkt"]).to_epsg() == 32632


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_item_selection_reads_only_selected_asset_window(
    two_band_item,
    pixel_reads,
    asynchronous,
    anyio_backend,
):
    """One selected asset and spatial window produce one full-resolution read."""
    if asynchronous:
        da = await lazycogs.open_item_async(
            two_band_item,
            bands=["b08", "b04"],
            store=LocalStore(),
        )
    else:
        da = lazycogs.open_item(two_band_item, bands=["b08", "b04"], store=LocalStore())
    assert pixel_reads == []
    assert list(da.band.values) == ["b08", "b04"]
    selected = da.sel(band="b08").isel(y=slice(17, 21), x=slice(23, 28))
    assert pixel_reads == []
    loaded = await selected.load_async() if asynchronous else selected.load()
    assert pixel_reads == [Window(col_off=23, row_off=17, width=5, height=4)]
    rows = np.arange(17, 21)[:, None]
    cols = np.arange(23, 28)[None, :]
    np.testing.assert_array_equal(
        loaded.to_numpy(),
        (cols + rows * 2048 + 1000) % 65535 + 1,
    )
    assert loaded.dims == ("y", "x")
    assert loaded.band.item() == "b08"
    assert loaded.dtype == np.dtype("uint16")
    assert loaded.attrs["_FillValue"] == 0


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_grid_mismatch_fails_before_pixel_reads(
    synthetic_cog,
    synthetic_cog_offgrid,
    pixel_reads,
    asynchronous,
):
    """Grid validation completes from headers before any pixel reads."""
    item = _item(b04=synthetic_cog, off=synthetic_cog_offgrid)
    if asynchronous:
        with pytest.raises(ValueError, match="native grid"):
            asyncio.run(
                lazycogs.open_item_async(
                    item,
                    bands=["b04", "off"],
                    store=LocalStore(),
                ),
            )
    else:
        with pytest.raises(ValueError, match="native grid"):
            lazycogs.open_item(item, bands=["b04", "off"], store=LocalStore())
    assert pixel_reads == []


def test_scalar_selection_reads_one_pixel(synthetic_cog, pixel_reads):
    """Scalar indexing returns a scalar with its spatial coordinates intact."""
    da = lazycogs.open_cog(synthetic_cog.as_uri(), store=LocalStore())
    selected = da.isel(band=0, y=17, x=23)
    assert pixel_reads == []
    assert selected.load().item() == (23 + 17 * 2048) % 65535 + 1
    assert pixel_reads == [Window(col_off=23, row_off=17, width=1, height=1)]
    assert selected.dims == ()
    assert selected.x.item() == da.x.values[23]
    assert selected.y.item() == da.y.values[17]


def test_item_selected_band_order_and_values(two_band_item, pixel_reads):
    """Reordered asset selections retain their labels and source values."""
    da = lazycogs.open_item(two_band_item, bands=["b04", "b08"], store=LocalStore())
    selected = da.isel(band=[1, 0], y=slice(17, 21), x=slice(23, 28))
    assert pixel_reads == []
    loaded = selected.load()
    assert len(pixel_reads) == 2
    assert all(
        w == Window(col_off=23, row_off=17, width=5, height=4) for w in pixel_reads
    )
    rows = np.arange(17, 21)[:, None]
    cols = np.arange(23, 28)[None, :]
    expected = np.stack(
        [(cols + rows * 2048 + seed) % 65535 + 1 for seed in [1000, 0]],
    )
    np.testing.assert_array_equal(loaded.to_numpy(), expected)
    assert list(loaded.band.values) == ["b08", "b04"]


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_sparse_repeated_item_bands(
    synthetic_cog,
    synthetic_cog_b,
    synthetic_cog_c,
    pixel_reads,
    asynchronous,
    anyio_backend,
):
    """Sparse duplicate selection reads each selected asset once, in output order."""
    item = _item(first=synthetic_cog, middle=synthetic_cog_b, last=synthetic_cog_c)
    if asynchronous:
        da = await lazycogs.open_item_async(
            item,
            bands=["first", "middle", "last"],
            store=LocalStore(),
        )
    else:
        da = lazycogs.open_item(
            item,
            bands=["first", "middle", "last"],
            store=LocalStore(),
        )
    assert pixel_reads == []
    selected = da.isel(band=[2, 0, 2], y=slice(17, 21), x=slice(23, 28))
    assert pixel_reads == []
    loaded = await selected.load_async() if asynchronous else selected.load()
    assert len(pixel_reads) == 2
    assert all(
        window == Window(col_off=23, row_off=17, width=5, height=4)
        for window in pixel_reads
    )
    rows = np.arange(17, 21)[:, None]
    cols = np.arange(23, 28)[None, :]
    expected = np.stack(
        [(cols + rows * 2048 + seed) % 65535 + 1 for seed in [2000, 0, 2000]],
    )
    np.testing.assert_array_equal(loaded.to_numpy(), expected)
    assert list(loaded.band.values) == ["last", "first", "last"]


@pytest.mark.anyio
async def test_sync_async_selected_results_match(two_band_item, anyio_backend):
    """Both entry points preserve values and coordinates after selection."""
    sync = lazycogs.open_item(two_band_item, bands=["b08", "b04"], store=LocalStore())
    async_da = await lazycogs.open_item_async(
        two_band_item,
        bands=["b08", "b04"],
        store=LocalStore(),
    )
    selection = {"band": [1, 0], "y": slice(7, 10), "x": slice(11, 15)}
    xr.testing.assert_identical(
        sync.isel(**selection).load(),
        await async_da.isel(**selection).load_async(),
    )


def test_open_item_rejects_item_without_assets():
    """An item with no assets raises a ValueError."""
    with pytest.raises(ValueError, match="no assets"):
        lazycogs.open_item({"assets": {}})
