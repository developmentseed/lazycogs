"""Tests for the lazycogs._explain module."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS
from rustac import DuckdbClient
from xarray.core import indexing

from lazycogs._backend import MultiBandStacBackendArray
from lazycogs._chunk_reader import _ChunkContext, _WindowContext
from lazycogs._explain import (
    ChunkRead,
    CogRead,
    ExplainPlan,
    _compute_chunk_bbox_4326,
    _find_backend_array,
    _infer_chunk_sizes,
    _inspect_item_async,
    _iter_spatial_chunks,
    _roi_pixel_offsets,
)
from lazycogs._mosaic_methods import FirstMethod
from lazycogs._temporal import _TimeStep

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wgs84() -> CRS:
    return CRS.from_epsg(4326)


@pytest.fixture
def epsg5070() -> CRS:
    return CRS.from_epsg(5070)


def _steps(filters: list[str]) -> list[_TimeStep]:
    """Return daily time steps for explain unit tests."""
    return [
        _TimeStep(
            coord=np.datetime64(value.split("/")[0], "D"),
            label=value,
            datetime_filter=value,
        )
        for value in filters
    ]


def _make_backend(
    crs: CRS,
    dates: list[str] | None = None,
    parquet_path: str = "/tmp/fake.parquet",
    bands: list[str] | None = None,
    dst_width: int = 10,
    dst_height: int = 10,
    affine: Affine | None = None,
) -> MultiBandStacBackendArray:
    """Return a minimal MultiBandStacBackendArray for unit testing."""
    if dates is None:
        dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02"]
    if bands is None:
        bands = ["red"]
    if affine is None:
        affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    return MultiBandStacBackendArray(
        parquet_path=parquet_path,
        duckdb_client=DuckdbClient(),
        bands=bands,
        time_steps=_steps(dates),
        dst_affine=affine,
        dst_crs=crs,
        bbox_4326=[0.0, 0.0, 10.0, 10.0],
        sortby=None,
        filter=None,
        ids=None,
        dst_width=dst_width,
        dst_height=dst_height,
        dtype=np.dtype("float32"),
        nodata=-9999.0,
        mosaic_method_cls=FirstMethod,
        dtype_was_explicit=True,
        nodata_was_explicit=True,
    )


def _make_da_with_backends(
    crs: CRS,
    dates: list[str],
    time_coords: list[np.datetime64],
    bands: list[str],
    width: int = 10,
    height: int = 10,
    affine: Affine | None = None,
) -> xr.DataArray:
    """Return a minimal lazycogs-backed DataArray for explain tests."""
    if affine is None:
        resolution = 1.0
        affine = Affine(resolution, 0.0, 0.0, 0.0, -resolution, float(height))

    backend = _make_backend(
        crs,
        dates=dates,
        bands=bands,
        dst_width=width,
        dst_height=height,
        affine=affine,
    )

    time_coord = np.array(time_coords)
    resolution = affine.a

    # Build coordinates matching the grid convention: x ascending, y descending
    x_coords = np.array([affine.c + (i + 0.5) * resolution for i in range(width)])
    y_coords = np.array([affine.f + (i + 0.5) * affine.e for i in range(height)])

    variable = xr.Variable(
        ("band", "time", "y", "x"),
        indexing.LazilyIndexedArray(backend),
    )
    return xr.DataArray(
        variable,
        coords={
            "band": bands,
            "time": time_coord,
            "y": y_coords,
            "x": x_coords,
        },
    )


# ---------------------------------------------------------------------------
# _iter_spatial_chunks
# ---------------------------------------------------------------------------


def test_iter_spatial_chunks_exact_fit():
    """A 4x4 grid with 2x2 chunks yields 4 tiles."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    tiles = list(_iter_spatial_chunks(affine, 4, 4, 2, 2))
    assert len(tiles) == 4
    rows = {t[0] for t in tiles}
    cols = {t[1] for t in tiles}
    assert rows == {0, 1}
    assert cols == {0, 1}
    # All tiles have full size
    for _, _, _, w, h in tiles:
        assert w == 2
        assert h == 2


def test_iter_spatial_chunks_edge_tiles():
    """A 10x10 grid with chunk=4 yields 3x3=9 tiles; edge tiles are smaller."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    tiles = list(_iter_spatial_chunks(affine, 10, 10, 4, 4))
    assert len(tiles) == 9  # ceil(10/4) = 3 in each dimension

    # Collect widths and heights by column/row
    col_widths = {}
    row_heights = {}
    for row, col, _, w, h in tiles:
        col_widths[col] = w
        row_heights[row] = h

    assert col_widths[0] == 4
    assert col_widths[1] == 4
    assert col_widths[2] == 2  # 10 - 8 = 2
    assert row_heights[0] == 4
    assert row_heights[1] == 4
    assert row_heights[2] == 2


def test_iter_spatial_chunks_single_tile():
    """When chunk >= extent, a single tile covering the whole area is yielded."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
    tiles = list(_iter_spatial_chunks(affine, 5, 5, 100, 100))
    assert len(tiles) == 1
    _, _, _, w, h = tiles[0]
    assert w == 5
    assert h == 5


def test_iter_spatial_chunks_affine_translation():
    """Tile affines are offset correctly from the ROI affine."""
    affine = Affine(2.0, 0.0, 10.0, 0.0, -2.0, 20.0)
    tiles = list(_iter_spatial_chunks(affine, 4, 4, 2, 2))
    assert len(tiles) == 4

    # First tile: same as ROI affine
    _, _, tile_affine_00, _, _ = tiles[0]
    assert tile_affine_00.c == pytest.approx(10.0)
    assert tile_affine_00.f == pytest.approx(20.0)

    # Second tile in x direction: offset by 2 pixels * 2 units/px = 4 units
    _, _, tile_affine_01, _, _ = tiles[1]
    assert tile_affine_01.c == pytest.approx(14.0)
    assert tile_affine_01.f == pytest.approx(20.0)

    # Second tile in y direction: offset by 2 pixels * 2 units/px = 4 units down
    _, _, tile_affine_10, _, _ = tiles[2]
    assert tile_affine_10.c == pytest.approx(10.0)
    assert tile_affine_10.f == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# _compute_chunk_bbox_4326
# ---------------------------------------------------------------------------


def test_compute_chunk_bbox_4326_wgs84(wgs84):
    """In EPSG:4326 the bbox is returned unchanged."""
    affine = Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0)
    bbox = _compute_chunk_bbox_4326(affine, 4, 1, wgs84)
    assert bbox == pytest.approx([10.0, 49.0, 14.0, 50.0])


def test_compute_chunk_bbox_4326_projected(epsg5070):
    """Projected CRS results are reprojected to WGS84."""
    affine = Affine(100.0, 0.0, 300000.0, 0.0, -100.0, 2700000.0)
    bbox = _compute_chunk_bbox_4326(affine, 10, 10, epsg5070)
    # Just verify it's a plausible lon/lat range for EPSG:5070 coordinates
    assert len(bbox) == 4
    minx, miny, maxx, maxy = bbox
    assert -180 <= minx < maxx <= 180
    assert -90 <= miny < maxy <= 90


# ---------------------------------------------------------------------------
# _infer_chunk_sizes
# ---------------------------------------------------------------------------


def test_infer_chunk_sizes_no_dask(wgs84):
    """Without dask the full spatial extent is returned as one chunk."""
    da = _make_da_with_backends(
        wgs84,
        dates=["2023-01-01/2023-01-01"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        bands=["red"],
        width=100,
        height=80,
    )
    chunk_h, chunk_w = _infer_chunk_sizes(da)
    assert chunk_h == 80
    assert chunk_w == 100


def test_infer_chunk_sizes_with_dask(wgs84):
    """With dask the first chunk size is used."""
    da = _make_da_with_backends(
        wgs84,
        dates=["2023-01-01/2023-01-01"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        bands=["red"],
        width=100,
        height=80,
    )
    da = da.chunk({"y": 32, "x": 64})
    chunk_h, chunk_w = _infer_chunk_sizes(da)
    assert chunk_h == 32
    assert chunk_w == 64


# ---------------------------------------------------------------------------
# _roi_pixel_offsets
# ---------------------------------------------------------------------------


def test_roi_pixel_offsets_full_extent(wgs84):
    """Full-extent DataArray yields zero offsets and full dimensions."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    da = _make_da_with_backends(
        wgs84,
        dates=["2023-01-01/2023-01-01"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        bands=["red"],
        width=10,
        height=10,
        affine=affine,
    )
    backend, _ = _find_backend_array(da.variable._data)
    x_start, y_start_physical, roi_w, roi_h = _roi_pixel_offsets(da, backend)
    assert x_start == 0
    assert y_start_physical == 0
    assert roi_w == 10
    assert roi_h == 10


# ---------------------------------------------------------------------------
# ExplainPlan
# ---------------------------------------------------------------------------


def _make_plan(n_bands: int = 2, n_time: int = 3, n_items: int = 1) -> ExplainPlan:
    """Return a minimal ExplainPlan for display tests."""
    chunk_reads = []
    for band in [f"band{i}" for i in range(n_bands)]:
        for t in range(n_time):
            items = [
                CogRead(
                    item_id=f"item-{t}-{j}",
                    asset_key=band,
                    href=f"s3://bucket/item-{t}-{j}.tif",
                )
                for j in range(n_items)
            ]
            chunk_reads.append(
                ChunkRead(
                    band=band,
                    time_index=t,
                    date_filter=f"2023-01-0{t + 1}/2023-01-0{t + 1}",
                    time_coord=np.datetime64(f"2023-01-0{t + 1}", "D"),
                    chunk_row=0,
                    chunk_col=0,
                    chunk_affine=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
                    chunk_width=10,
                    chunk_height=10,
                    cog_reads=items,
                ),
            )
    return ExplainPlan(
        href="/tmp/fake.parquet",
        crs="EPSG:4326",
        resolution=1.0,
        bands=[f"band{i}" for i in range(n_bands)],
        time_coords=[np.datetime64(f"2023-01-0{t + 1}", "D") for t in range(n_time)],
        dst_width=10,
        dst_height=10,
        chunk_width=10,
        chunk_height=10,
        chunk_reads=chunk_reads,
        fetch_headers=False,
    )


def test_explain_plan_repr():
    """__repr__ renders without error and contains key counts."""
    plan = _make_plan(n_bands=2, n_time=3, n_items=1)
    r = repr(plan)
    assert "2 band(s)" in r
    assert "3 time step(s)" in r
    assert "6 chunk read(s)" in r


def test_explain_plan_summary():
    """summary() renders without error and contains expected sections."""
    plan = _make_plan(n_bands=1, n_time=2, n_items=2)
    s = plan.summary()
    assert "ExplainPlan" in s
    assert "EPSG:4326" in s
    assert "band0" in s


def test_explain_plan_summary_with_empty_chunks():
    """summary() correctly counts empty chunks."""
    plan = _make_plan(n_bands=1, n_time=2, n_items=0)
    s = plan.summary()
    assert "2" in s  # 2 empty chunks


def test_explain_plan_summary_fetch_headers():
    """summary() includes overview and window stats when fetch_headers=True."""
    chunk_reads = [
        ChunkRead(
            band="red",
            time_index=0,
            date_filter="2023-01-01/2023-01-01",
            time_coord=np.datetime64("2023-01-01", "D"),
            chunk_row=0,
            chunk_col=0,
            chunk_affine=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
            chunk_width=10,
            chunk_height=10,
            cog_reads=[
                CogRead(
                    item_id="item-0",
                    asset_key="red",
                    href="s3://bucket/item-0.tif",
                    overview_level=1,
                    overview_resolution=20.0,
                    window_col_off=0,
                    window_row_off=0,
                    window_width=256,
                    window_height=256,
                ),
            ],
        ),
    ]
    plan = ExplainPlan(
        href="/tmp/fake.parquet",
        crs="EPSG:4326",
        resolution=1.0,
        bands=["red"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        dst_width=10,
        dst_height=10,
        chunk_width=10,
        chunk_height=10,
        chunk_reads=chunk_reads,
        fetch_headers=True,
    )
    s = plan.summary()
    assert "ovr 1" in s
    assert "256 x 256" in s
    assert "fetch_headers=True" not in s  # no hint to enable it


def test_explain_plan_to_dataframe():
    """to_dataframe() returns correct columns and row count."""
    pytest.importorskip("pandas")
    plan = _make_plan(n_bands=2, n_time=3, n_items=2)
    df = plan.to_dataframe()
    expected_cols = {
        "band",
        "time_index",
        "date_filter",
        "chunk_row",
        "chunk_col",
        "n_cog_reads",
        "item_id",
        "href",
        "overview_level",
        "window_width",
    }
    assert expected_cols.issubset(df.columns)
    # 2 bands * 3 time steps * 2 items each = 12 rows
    assert len(df) == 12


def test_explain_plan_to_dataframe_empty_chunks():
    """to_dataframe() includes one row per empty chunk."""
    pytest.importorskip("pandas")
    plan = _make_plan(n_bands=1, n_time=2, n_items=0)
    df = plan.to_dataframe()
    assert len(df) == 2
    assert df["item_id"].isna().all()


def test_explain_plan_properties():
    """total_chunk_reads, total_cog_reads, empty_chunk_count are correct."""
    plan = _make_plan(n_bands=2, n_time=3, n_items=1)
    assert plan.total_chunk_reads == 6
    assert plan.total_cog_reads == 6
    assert plan.empty_chunk_count == 0

    empty_plan = _make_plan(n_bands=1, n_time=2, n_items=0)
    assert empty_plan.empty_chunk_count == 2
    assert empty_plan.total_cog_reads == 0


# ---------------------------------------------------------------------------
# StacCogAccessor.explain() via mocked rustac
# ---------------------------------------------------------------------------


def _fake_items(band: str, n: int) -> list[dict]:
    return [
        {
            "id": f"item-{i}",
            "assets": {
                band: {"href": f"s3://bucket/item-{i}.tif"},
            },
        }
        for i in range(n)
    ]


def test_inspect_item_async_builds_window_context_for_header_fetch(wgs84):
    """Header fetch explain path does not require pixel-read contract fields."""
    chunk_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    store = object()
    geotiff = SimpleNamespace(overviews=[], transform=chunk_affine)
    reader = geotiff
    window = SimpleNamespace(col_off=1, row_off=2, width=3, height=4)

    with patch(
        "lazycogs._explain._open_and_window",
        new_callable=AsyncMock,
        return_value=(geotiff, reader, window, "s3://bucket/item-0.tif"),
    ) as open_and_window:
        read = asyncio.run(
            _inspect_item_async(
                _fake_items("red", 1)[0],
                "red",
                chunk_affine,
                wgs84,
                10,
                10,
                store,
            ),
        )

    _, _, ctx = open_and_window.call_args.args
    assert isinstance(ctx, _WindowContext)
    assert not isinstance(ctx, _ChunkContext)
    assert ctx.chunk_affine == chunk_affine
    assert ctx.dst_crs == wgs84
    assert ctx.chunk_width == 10
    assert ctx.chunk_height == 10
    assert ctx.store is store
    assert ctx.path_fn is None
    assert not hasattr(ctx, "out_dtype")
    assert not hasattr(ctx, "nodata")
    assert read is not None
    assert read.window_col_off == 1
    assert read.window_row_off == 2
    assert read.window_width == 3
    assert read.window_height == 4


def test_accessor_raises_on_non_stac_da():
    """explain() raises ValueError when the array is not lazycogs-backed."""
    da = xr.DataArray(np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"backed by lazycogs"):
        da.lazycogs.explain()


def test_accessor_explain_routes_duckdb_queries_through_helper(wgs84):
    """explain() routes DuckDB work through the shared helper."""
    dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02"]
    time_coords = [np.datetime64("2023-01-01", "D"), np.datetime64("2023-01-02", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )

    with patch(
        "lazycogs._explain.run_duckdb",
        new_callable=AsyncMock,
        return_value=_fake_items("red", 2),
    ) as mock_run_duckdb:
        plan = da.lazycogs.explain()

    assert plan.total_chunk_reads == 2
    assert plan.total_cog_reads == 4
    assert mock_run_duckdb.await_count == 2


def test_accessor_explain_uses_backend_time_step_filter_with_subdaily_coords(wgs84):
    """Explain uses _TimeStep predicates and preserves sub-daily coordinates."""
    step = _TimeStep(
        coord=np.datetime64("2023-01-01T01:00:00", "s"),
        label="2023-01-01T01:00:00Z",
        datetime_filter="2023-01-01T01:00:00Z/2023-01-01T01:59:59Z",
    )
    backend = MultiBandStacBackendArray(
        parquet_path="/tmp/fake.parquet",
        duckdb_client=DuckdbClient(),
        bands=["red"],
        time_steps=[step],
        dst_affine=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        dst_crs=wgs84,
        bbox_4326=[0.0, 0.0, 4.0, 4.0],
        sortby=None,
        filter="eo:cloud_cover < 20",
        ids=None,
        dst_width=4,
        dst_height=4,
        dtype=np.dtype("float32"),
        nodata=-9999.0,
        mosaic_method_cls=FirstMethod,
        dtype_was_explicit=True,
        nodata_was_explicit=True,
    )
    variable = xr.Variable(
        ("band", "time", "y", "x"),
        indexing.LazilyIndexedArray(backend),
    )
    da = xr.DataArray(
        variable,
        coords={
            "band": ["red"],
            "time": np.array([step.coord]),
            "y": np.array([3.5, 2.5, 1.5, 0.5]),
            "x": np.array([0.5, 1.5, 2.5, 3.5]),
        },
    )

    with patch("rustac.DuckdbClient.search", return_value=[]) as search:
        plan = da.lazycogs.explain()

    assert plan.time_coords == [step.coord]
    assert plan.chunk_reads[0].date_filter == step.datetime_filter
    assert search.call_args.kwargs["datetime"] == step.datetime_filter
    assert search.call_args.kwargs["filter"] == "eo:cloud_cover < 20"


def test_accessor_explain_returns_plan(wgs84):
    """explain() returns an ExplainPlan with correct counts."""
    dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02"]
    time_coords = [np.datetime64("2023-01-01", "D"), np.datetime64("2023-01-02", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = _fake_items("red", 2)
        plan = da.lazycogs.explain()

    # 1 band * 2 time steps * 1 spatial tile (no chunking) = 2 chunk reads
    assert plan.total_chunk_reads == 2
    assert plan.total_cog_reads == 4  # 2 items per chunk * 2 chunks
    assert plan.bands == ["red"]
    assert len(plan.time_coords) == 2


def test_accessor_explain_empty_results(wgs84):
    """explain() handles chunks with zero matching items gracefully."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da.lazycogs.explain()

    assert plan.total_chunk_reads == 1
    assert plan.total_cog_reads == 0
    assert plan.empty_chunk_count == 1


def test_accessor_explain_with_dask_chunks(wgs84):
    """explain() uses dask chunk sizes when available."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=8,
        height=8,
    )
    da = da.chunk({"y": 4, "x": 4})

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da.lazycogs.explain()

    # 1 band * 1 time * 4 spatial tiles (2x2 grid from 8px / 4px chunks)
    assert plan.total_chunk_reads == 4
    assert plan.chunk_width == 4
    assert plan.chunk_height == 4


def test_accessor_explain_multiple_bands(wgs84):
    """explain() iterates over all active bands."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red", "green", "blue"],
        width=4,
        height=4,
    )

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = _fake_items("red", 1)
        plan = da.lazycogs.explain()

    # 3 bands * 1 time * 1 spatial tile = 3 chunk reads
    assert plan.total_chunk_reads == 3
    assert set(plan.bands) == {"red", "green", "blue"}


def test_accessor_explain_band_slice(wgs84):
    """explain() on a single-band slice only queries that band's backend."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red", "green"],
        width=4,
        height=4,
    )
    da_red = da.sel(band="red")

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da_red.lazycogs.explain()

    assert plan.bands == ["red"]
    assert plan.total_chunk_reads == 1  # only 1 band


def test_accessor_explain_time_slice(wgs84):
    """explain() on a time-sliced DataArray only queries matching time steps."""
    dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02", "2023-01-03/2023-01-03"]
    time_coords = [
        np.datetime64("2023-01-01", "D"),
        np.datetime64("2023-01-02", "D"),
        np.datetime64("2023-01-03", "D"),
    ]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )
    da_sliced = da.isel(time=slice(0, 2))  # first 2 time steps

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da_sliced.lazycogs.explain()

    assert len(plan.time_coords) == 2
    assert plan.total_chunk_reads == 2  # 1 band * 2 time steps * 1 spatial tile


def test_accessor_explain_chunk_then_sel_time(wgs84):
    """explain() works after .chunk(...).sel(time=...), a dask getitem on top.

    Regression test: chunking before selecting a single time label builds the
    dask graph with the time selection applied as a separate ``getitem``
    layer rather than folded into the discovered backend's indexer key, so
    explain() must not rely on recovering that key to figure out which
    backend time step is active.
    """
    dates = [
        "2023-01-01/2023-01-01",
        "2023-01-02/2023-01-02",
        "2023-01-03/2023-01-03",
    ]
    time_coords = [
        np.datetime64("2023-01-01", "D"),
        np.datetime64("2023-01-02", "D"),
        np.datetime64("2023-01-03", "D"),
    ]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=8,
        height=8,
    )
    da_selected = da.chunk(x=4, y=4).sel(time="2023-01-02")

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da_selected.lazycogs.explain()

    assert plan.time_coords == [np.datetime64("2023-01-02", "D")]
    assert [chunk.date_filter for chunk in plan.chunk_reads] == [
        "2023-01-02/2023-01-02",
    ] * plan.total_chunk_reads


def test_accessor_explain_time_sort_preserves_current_order(wgs84):
    """explain() follows the current DataArray time order after sortby."""
    # Backend time steps are stored out of chronological order; the
    # DataArray's "time" coordinate always matches each step's own coord.
    dates = ["2023-01-02/2023-01-02", "2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-02", "D"), np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )
    da_sorted = da.sortby("time")

    with patch("rustac.DuckdbClient.search", return_value=[]):
        plan = da_sorted.lazycogs.explain()

    assert plan.time_coords == [
        np.datetime64("2023-01-01", "D"),
        np.datetime64("2023-01-02", "D"),
    ]
    assert [chunk.date_filter for chunk in plan.chunk_reads] == [
        "2023-01-01/2023-01-01",
        "2023-01-02/2023-01-02",
    ]


def test_accessor_explain_query_count_not_multiplied_by_bands(wgs84):
    """DuckDB is queried once per (time, tile) — not once per (band, time, tile)."""
    dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02"]
    time_coords = [np.datetime64("2023-01-01", "D"), np.datetime64("2023-01-02", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red", "green", "blue"],
        width=4,
        height=4,
    )

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = _fake_items("red", 1)
        plan = da.lazycogs.explain()

    # 3 bands * 2 time steps * 1 spatial tile = 6 chunk reads
    assert plan.total_chunk_reads == 6
    # but only 2 DuckDB queries (T=2 x S=1), not 6 (B*T*S)
    assert mock_search.call_count == 2
