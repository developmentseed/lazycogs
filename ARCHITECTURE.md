# Architecture: lazycogs

lazycogs turns a geoparquet STAC item index into a lazy `(band, time, y, x)` xarray `DataArray` backed by Cloud-Optimized GeoTIFFs. Raster I/O is handled by `async-geotiff`; STAC filtering is handled by `rustac` and DuckDB; reprojection is implemented with `pyproj` and numpy.

## Module map

```text
src/lazycogs/
  _core.py           open(), representative-item inspection, dtype/nodata resolution, time-step building, DataArray assembly
  _backend.py        MultiBandStacBackendArray, xarray indexing bridge, per-chunk orchestration
  _chunk_reader.py   async COG opening, overview/window selection, reads, reprojection, mosaicking
  _executor.py       shared background event loop and bounded executors
  _explain.py        dry-run read estimator exposed as da.lazycogs.explain()
  _grid.py           output affine transform and grid dimensions
  _reproject.py      warp-map computation and nearest-neighbor sampling
  _storage_ext.py    STAC Storage Extension metadata parsing
  _single.py         open_cog()/open_item(): native-resolution single-COG and single-item reads
  _store.py          HREF-to-store resolution and store_for()
  _temporal.py       temporal grouping strategies and _TimeStep predicates
  _mosaic_methods.py pixel-selection strategies
```

## STAC query contract

`open()` accepts a local geoparquet file or, when a custom `DuckdbClient` is supplied, a parquet directory such as a hive-partitioned archive. It does not query a remote STAC API directly. The intended workflow is to run `rustac.search_to("items.parquet", api_url, ...)` once, then pass that parquet source to lazycogs. This keeps dask chunk execution from turning into hundreds of concurrent STAC API calls.

When `duckdb_client=None`, `open()` creates a plain `DuckdbClient()` and requires a `.parquet` or `.geoparquet` path. A caller-supplied client is used for every startup and per-chunk query, including clients configured for hive partition pruning.

## Two-phase execution model

### Phase 0: open time

`open()` does only enough eager work to return a fully described lazy array:

1. Resolve the DuckDB client and validate the source path contract.
2. Parse `time_period` into a temporal grouper.
3. Transform the requested bbox into EPSG:4326 for STAC spatial filtering.
4. Query one representative item and open one COG per selected band to infer the output dtype/nodata contract.
5. Query Arrow-backed datetime columns and build sorted `_TimeStep` objects. Each time step carries the exact `rustac` `datetime=` predicate used later during reads.
6. Compute the output grid.
7. Build one `MultiBandStacBackendArray` with shape `(band, time, y, x)` and wrap it in an xarray `LazilyIndexedArray`, or chunk it only when `chunks` is explicitly provided.
8. Attach spatial coordinates and metadata. `spatial:transform` uses affine coefficient order; `spatial_ref.attrs["GeoTransform"]` uses GDAL geotransform order.

No pixel windows are read at open time. Runtime state such as the DuckDB client and store is kept on the backend array, not in `DataArray.attrs`, so xarray copies do not try to pickle live resources.

### Phase 1: compute time

When xarray asks the backend array for data, `MultiBandStacBackendArray.__getitem__` resolves the `(band, time, y, x)` key into a concrete chunk:

1. Resolve band and time indices, including dimension squeezes for scalar keys.
2. Convert scalar y/x selections into size-1 slices and compute the chunk affine transform.
3. Compute the chunk bbox in EPSG:4326.
4. Run one DuckDB spatial query per selected time step.
5. For each non-empty time step, call `read_chunk_async()` to read all selected bands from the matching COGs and feed them to the mosaic method.
6. Return data in top-down raster order.

The sync path submits this coroutine to the shared lazycogs event loop via `run_on_loop()`. The async path uses xarray's async indexing adapter and stays on the caller's loop.

## Grid and coordinate convention

`compute_output_grid()` creates a north-up raster grid with an affine transform whose origin is the top-left corner of the top-left pixel and whose y scale is negative. Pixel-centre x coordinates increase west-to-east; y coordinates decrease north-to-south. Label-based y slicing therefore uses `sel(y=slice(north, south))`.

A `rasterix.RasterIndex` is attached to every returned array for CRS discovery and spatial alignment. The x/y coordinate variables are materialised as eager numpy arrays so chunked scalar spatial selections remain scalars after compute.

## Per-chunk read pipeline

Each item-band read validates the source COG against the resolved output contract, then follows this pipeline:

1. **Overview selection.** `_select_overview()` picks the coarsest overview whose pixel size is still no larger than the target pixel size in the source CRS. Full resolution is used when no overview avoids upsampling.
2. **Source window computation.** The destination chunk bbox is transformed to the source CRS, mapped through the inverse source affine, clamped to image bounds, and skipped if it does not overlap.
3. **Tile read.** `async-geotiff` reads the selected window from the chosen overview or full-resolution image.
4. **Warp-map reprojection.** `_apply_bands_with_warp_cache()` uses `compute_warp_map()` and `apply_warp_map()` to nearest-neighbor sample source pixels onto the destination chunk grid. The cache key is `(tuple(raster.transform), src_crs)` so bands or time steps with identical source geometry can reuse the same map.
5. **Mosaic.** `_drain_in_order()` feeds completed reads to the mosaic method in source order even though network reads may complete out of order.

Nearest-neighbor is the only supported resampling method. The STAC projection extension cannot replace COG header reads because overview transforms, dimensions, and tile byte ranges still come from the COG IFD chain.

Read failures are controlled by `errors=`. `errors="raise"` wraps non-contract failures in `ChunkReadError` with `item_id`, `bands`, and the original exception. `errors="ignore"` logs a warning and leaves the mosaic fill value for that item's pixels. Dtype and nodata contract violations always raise.

## Explain plans

`da.lazycogs.explain()` discovers the underlying `MultiBandStacBackendArray` from a still-lazy DataArray and runs the same DuckDB spatial queries needed for the current view. It issues one query per `(time step, spatial tile)` combination and fans those results across active bands in Python. It stops before pixel I/O.

With `fetch_headers=True`, explain also opens COG headers and populates `CogRead.overview_level`, window offsets, and window sizes. That path uses `_chunk_reader._open_and_window()` with a lightweight `_WindowContext`, shared with real reads, without constructing the full pixel-read context.

## Concurrency model

lazycogs uses three layers of concurrency:

- **Dask chunk scheduling** is optional. If the array is dask-backed, dask decides how many chunk tasks run at once.
- **One shared asyncio loop** handles COG header reads, window reads, per-time-step fan-out, and the sync bridge. The loop is persistent for the lifetime of the process because async storage libraries can retain loop-bound resources and callbacks after an awaited read returns.
- **Bounded executors** keep blocking work off the event loop. DuckDB calls go through `run_duckdb()` and a single-worker executor. CPU-heavy reprojection work runs in a shared thread pool sized by `LAZYCOGS_REPROJECT_WORKERS` before first use, defaulting to `min(os.cpu_count() or 1, 4)`.

Threads are used for reprojection because `pyproj.Transformer.transform()` and numpy's indexing kernels release the GIL during heavy work, avoiding process-pool serialization and array-pickling overhead. `compute_warp_map()` and `apply_warp_map()` are usually memory-bandwidth-bound, so increasing the thread count far beyond the default often hurts more than it helps.

## Chunking strategy

The async read layer already fans out COG I/O within one chunk. Spatial dask chunks add more DuckDB queries, more COG opens, and more smaller gathers. Use spatial chunks only when the full array cannot fit in memory, and make them as large as memory allows.

Recommended defaults:

| Goal | Suggested `chunks` |
|---|---|
| Maximum throughput, array fits in memory | omit `chunks` or pass `chunks={}` |
| Array too large for memory | `{"time": 1, "x": N, "y": N}` with large spatial chunks |

Band chunking does not help because all selected bands are read together per item. Time chunking can help when dask should run multiple time chunks in parallel, but non-dask reads already parallelise selected time steps inside one event loop gather.

## Temporal grouping

`_temporal.py` converts `time_period` into `_TimeStep` objects. `_TimeStep.coord` is the xarray coordinate; `_TimeStep.datetime_filter` is passed directly to `rustac` during chunk reads.

| `time_period` | Grouping |
|---|---|
| `None` | unique normalized timestamp |
| `PTnH` | fixed hour windows aligned to 2000-01-01T00:00:00Z |
| `P1D` | calendar day |
| `P1W` | ISO calendar week |
| `P1M` | calendar month |
| `P1Y` | calendar year |
| `PnD` / `PnW` where n > 1 | fixed day windows aligned to 2000-01-01 |

Exact and hourly grouping reject bare dates because `YYYY-MM-DD` is a day-wide `rustac` predicate, not an exact instant.

## Mosaic and temporal compositing

Mosaic methods operate on nodata masks only. `FirstMethod` returns the first valid pixel and can stop early once the output is filled. `HighestMethod`, `LowestMethod`, `MeanMethod`, `MedianMethod`, `StdevMethod`, and `CountMethod` implement other pure-numpy reductions. Float-only methods advertise `requires_float=True`; `open()` auto-promotes inferred integer outputs to `float32` and rejects explicit integer dtypes for those methods.

Combining `time_period` with a mosaic method is the preferred way to build temporal composites. For example, `time_period="P1W"` groups all items in each ISO week into one output time step, and the mosaic method receives only that week's items. Post-hoc xarray reductions over a daily array usually require materialising every time step first.

## Store resolution

`open(..., store=...)` forwards any caller-supplied `async_geotiff.Store` to `GeoTIFF.open()`. When `store=None`, `_store.resolve()` delegates scheme handling to `obstore.store.from_url()` and caches the resulting store per root URL (`scheme://netloc`) behind a lock.

No credential defaults are applied by lazycogs. Obstore's own environment-based credential discovery is used. For public buckets, authenticated access, requester-pays buckets, custom endpoints, or unusual URL layouts, construct a store explicitly and pass it to `open()`. `path_from_href` can override object-path extraction when the store root does not match the asset URL structure.

`store_for()` is a public convenience factory that samples one STAC item from a geoparquet source, derives the data asset root URL, applies supported STAC Storage Extension hints, and returns a fresh uncached obstore `ObjectStore`.

## Documentation surfaces

Keep durable architecture invariants here. Keep user-facing usage in `README.md` and the guides. Keep benchmark numbers, one-off design rebuttals, and implementation-history notes in `dev-docs/` or the changelog rather than in this file.
