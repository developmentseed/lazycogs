"""Tests for lazycogs._native_grid."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from affine import Affine

from lazycogs._native_grid import NativeGrid, _snap_bbox, native_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    item_id: str = "item-0",
    epsg: int = 32612,
    red_transform: list[float] | None = None,
    nir_transform: list[float] | None = None,
) -> dict:
    """Build a minimal STAC item dict with projection extension fields."""
    if red_transform is None:
        red_transform = [10, 0, 699960, 0, -10, 4200000]
    assets: dict = {
        "red": {
            "href": f"file:///cogs/{item_id}/red.tif",
            "type": "image/tiff; application=geotiff",
            "roles": ["data"],
            "proj:shape": [10980, 10980],
            "proj:transform": red_transform,
        }
    }
    if nir_transform is not None:
        assets["nir08"] = {
            "href": f"file:///cogs/{item_id}/nir08.tif",
            "type": "image/tiff; application=geotiff",
            "roles": ["data"],
            "proj:shape": [5490, 5490],
            "proj:transform": nir_transform,
        }
    return {
        "id": item_id,
        "properties": {"datetime": "2025-07-04T00:00:00Z", "proj:epsg": epsg},
        "assets": assets,
    }


def _mock_duckdb_client(items: list[dict]) -> MagicMock:
    """Return a DuckdbClient mock whose search() always returns *items*."""
    client = MagicMock()
    client.search.return_value = items
    return client


# ---------------------------------------------------------------------------
# _snap_bbox
# ---------------------------------------------------------------------------


def test_snap_bbox_aligned_input():
    """A bbox already on the pixel grid is returned unchanged."""
    transform = Affine(10.0, 0.0, 699960.0, 0.0, -10.0, 4200000.0)
    bbox = _snap_bbox(709960.0, 4168000.0, 739960.0, 4198000.0, transform)
    assert bbox == pytest.approx((709960.0, 4168000.0, 739960.0, 4198000.0))


def test_snap_bbox_expands_outward():
    """A non-aligned bbox is snapped outward to fully contain the input."""
    transform = Affine(10.0, 0.0, 699960.0, 0.0, -10.0, 4200000.0)
    # Slightly inside grid boundaries on all sides.
    bbox = _snap_bbox(709965.0, 4168005.0, 739955.0, 4197995.0, transform)
    snapped_minx, snapped_miny, snapped_maxx, snapped_maxy = bbox
    assert snapped_minx <= 709965.0
    assert snapped_miny <= 4168005.0
    assert snapped_maxx >= 739955.0
    assert snapped_maxy >= 4197995.0


def test_snap_bbox_boundaries_on_grid():
    """Snapped corners always land exactly on pixel boundaries."""
    transform = Affine(10.0, 0.0, 699960.0, 0.0, -10.0, 4200000.0)
    bbox = _snap_bbox(709963.0, 4168007.0, 739957.0, 4197992.0, transform)
    ox, px = transform.c, transform.a
    oy, py = transform.f, transform.e
    minx, miny, maxx, maxy = bbox
    assert math.isclose((minx - ox) % px, 0.0, abs_tol=1e-6)
    assert math.isclose((maxx - ox) % px, 0.0, abs_tol=1e-6)
    assert math.isclose((maxy - oy) % py, 0.0, abs_tol=1e-6)
    assert math.isclose((miny - oy) % py, 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# native_grid
# ---------------------------------------------------------------------------


def test_native_grid_returns_correct_crs_and_resolution():
    """CRS and resolution are extracted from proj:epsg and proj:transform."""
    item = _make_item(epsg=32612, red_transform=[10, 0, 699960, 0, -10, 4200000])
    client = _mock_duckdb_client([item])
    result = native_grid(
        "fake.parquet",
        bbox=(-108.5, 37.5, -107.5, 38.5),
        bands=["red"],
        duckdb_client=client,
    )
    assert result.crs == "EPSG:32612"
    assert result.resolution == 10.0


def test_native_grid_bbox_is_in_native_crs():
    """The returned bbox is in the native CRS (larger coords than WGS-84 degrees)."""
    item = _make_item(epsg=32612)
    client = _mock_duckdb_client([item])
    result = native_grid(
        "fake.parquet",
        bbox=(-108.5, 37.5, -107.5, 38.5),
        bands=["red"],
        duckdb_client=client,
    )
    minx, miny, maxx, maxy = result.bbox
    # UTM coordinates are in metres, so values should be much larger than degrees.
    assert minx > 10_000
    assert miny > 10_000


def test_native_grid_bbox_snapped_to_grid():
    """The returned bbox corners fall on the native pixel grid."""
    transform_list = [10, 0, 699960, 0, -10, 4200000]
    item = _make_item(red_transform=transform_list)
    client = _mock_duckdb_client([item])
    result = native_grid(
        "fake.parquet",
        bbox=(-108.5, 37.5, -107.5, 38.5),
        bands=["red"],
        duckdb_client=client,
    )
    ox, px = 699960.0, 10.0
    oy, py = 4200000.0, -10.0
    minx, miny, maxx, maxy = result.bbox
    assert math.isclose((minx - ox) % px, 0.0, abs_tol=1e-3)
    assert math.isclose((maxx - ox) % px, 0.0, abs_tol=1e-3)
    assert math.isclose((maxy - oy) % py, 0.0, abs_tol=1e-3)
    assert math.isclose((miny - oy) % py, 0.0, abs_tol=1e-3)


def test_native_grid_bbox_contains_projected_input():
    """The snapped bbox contains the projected version of the input bbox."""
    from pyproj import Transformer

    item = _make_item(epsg=32612)
    client = _mock_duckdb_client([item])
    input_bbox = (-108.5, 37.5, -107.5, 38.5)
    result = native_grid(
        "fake.parquet", bbox=input_bbox, bands=["red"], duckdb_client=client
    )
    t = Transformer.from_crs(4326, 32612, always_xy=True)
    xs, ys = t.transform(
        [input_bbox[0], input_bbox[2], input_bbox[0], input_bbox[2]],
        [input_bbox[1], input_bbox[1], input_bbox[3], input_bbox[3]],
    )
    assert result.bbox[0] <= min(xs)
    assert result.bbox[1] <= min(ys)
    assert result.bbox[2] >= max(xs)
    assert result.bbox[3] >= max(ys)


def test_native_grid_no_items_raises():
    """A ValueError is raised when the parquet query returns no items."""
    client = _mock_duckdb_client([])
    with pytest.raises(ValueError, match="No STAC items found"):
        native_grid("fake.parquet", bbox=(-1.0, -1.0, 1.0, 1.0), duckdb_client=client)


def test_native_grid_mixed_crs_raises():
    """Items spanning multiple CRSs raise a ValueError."""
    item_a = _make_item("a", epsg=32612)
    item_b = _make_item("b", epsg=32613)
    client = _mock_duckdb_client([item_a, item_b])
    with pytest.raises(ValueError, match="multiple CRSs"):
        native_grid(
            "fake.parquet", bbox=(-110.0, 37.0, -105.0, 39.0), duckdb_client=client
        )


def test_native_grid_missing_proj_transform_raises():
    """An asset without proj:transform raises a descriptive ValueError."""
    item = _make_item()
    del item["assets"]["red"]["proj:transform"]
    client = _mock_duckdb_client([item])
    with pytest.raises(ValueError, match="proj:transform"):
        native_grid(
            "fake.parquet", bbox=(-108.5, 37.5, -107.5, 38.5), duckdb_client=client
        )


def test_native_grid_missing_band_raises():
    """Requesting a band absent from all items raises a ValueError."""
    item = _make_item()
    client = _mock_duckdb_client([item])
    with pytest.raises(ValueError, match="not found"):
        native_grid(
            "fake.parquet",
            bbox=(-108.5, 37.5, -107.5, 38.5),
            bands=["scl"],
            duckdb_client=client,
        )


def test_native_grid_mixed_resolution_raises():
    """Bands with different native resolutions raise a ValueError."""
    item = _make_item(
        red_transform=[10, 0, 699960, 0, -10, 4200000],
        nir_transform=[20, 0, 699960, 0, -20, 4200000],
    )
    client = _mock_duckdb_client([item])
    with pytest.raises(ValueError, match="different native resolutions"):
        native_grid(
            "fake.parquet",
            bbox=(-108.5, 37.5, -107.5, 38.5),
            bands=["red", "nir08"],
            duckdb_client=client,
        )


def test_native_grid_infers_bands_from_first_item():
    """When bands=None, data assets from the first item are used."""
    item = _make_item(
        red_transform=[10, 0, 699960, 0, -10, 4200000],
    )
    client = _mock_duckdb_client([item])
    result = native_grid(
        "fake.parquet", bbox=(-108.5, 37.5, -107.5, 38.5), duckdb_client=client
    )
    assert result.resolution == 10.0


def test_native_grid_nine_element_transform():
    """A 9-element proj:transform (with the identity last row) is handled."""
    item = _make_item(red_transform=[10, 0, 699960, 0, -10, 4200000, 0, 0, 1])
    client = _mock_duckdb_client([item])
    result = native_grid(
        "fake.parquet", bbox=(-108.5, 37.5, -107.5, 38.5), duckdb_client=client
    )
    assert result.resolution == 10.0


def test_native_grid_result_is_frozen():
    """NativeGrid instances are immutable."""
    grid = NativeGrid(crs="EPSG:32612", resolution=10.0, bbox=(0.0, 0.0, 1.0, 1.0))
    with pytest.raises((AttributeError, TypeError)):
        grid.crs = "EPSG:4326"  # type: ignore[misc]
