<p align="center">
  <img src="https://raw.githubusercontent.com/developmentseed/lazycogs/main/lazycogs.png" alt="lazycogs">
</p>

[![CI](https://github.com/developmentseed/lazycogs/actions/workflows/ci.yml/badge.svg)](https://github.com/developmentseed/lazycogs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lazycogs)](https://pypi.org/project/lazycogs/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lazycogs)](https://pypi.org/project/lazycogs/)
[![License](https://img.shields.io/github/license/developmentseed/lazycogs)](https://github.com/developmentseed/lazycogs/blob/main/LICENSE)

Open a lazy `(band, time, y, x)` xarray DataArray from thousands of cloud-optimized geotiffs (COGs). No GDAL required.

## What is lazycogs?

[stackstac](https://stackstac.readthedocs.io) and [odc-stac](https://odc-stac.readthedocs.io) established the pattern that lazycogs builds on: take a STAC item collection and expose it as a spatially-aligned xarray DataArray ready for dask-parallel computation. Both are excellent tools that cover most satellite imagery workflows well. They rely on the trusty combination of rasterio and GDAL for data i/o and warping operations.

lazycogs takes the same approach but replaces GDAL and rasterio with a Rust-native stack: [rustac](https://stac-utils.github.io/rustac-py) for STAC queries over stac-geoparquet files, [async-geotiff](https://developmentseed/async-geotiff) for COG i/o, and [obstore](https://developmentseed.org/obstore) as the default cloud storage integration.

The result is a tool that can instantly expose a lazy xarray DataArray view of massive STAC item archives in any CRS and resolution. Each array operation triggers a targeted spatial query on the stac-geoparquet file to find only the assets needed for that specific chunk — no upfront scan of every item required.

One constraint worth naming: lazycogs only reads Cloud Optimized GeoTIFFs. If your assets are in another format, this is not the right tool.

| Task | Library |
|---|---|
| STAC search + spatial indexing | `rustac` (DuckDB + geoparquet) |
| COG I/O | `async-geotiff` (Rust, no GDAL) |
| Cloud storage | `obstore` by default; any `async_geotiff.Store` when passed via `store=` |
| Reprojection | `pyproj` + numpy |
| Lazy dataset construction | xarray `BackendEntrypoint` + `LazilyIndexedArray` |

## Installation

```bash
pip install lazycogs
```

## Coordinate convention

`lazycogs.open()` returns a DataArray whose `y` coordinates follow the standard
north-up raster convention with the origin in the top left (not bottom left). That is, `y` coordinates are **descending** from north to south.  In other words,
y label `0` is the northernmost pixel and `y[-1]` is the southernmost.  This
matches the affine transform and is consistent with `odc-stac`, `rioxarray`, and
GDAL.

Use ``sel(y=slice(north, south))`` (high to low) for spatial subsetting.

`x` and `y` keep their `RasterIndex`-based spatial selection behavior, but the
coordinate variables themselves are materialized eagerly so chunked nearest-neighbor
spatial selections compute cleanly.

## Example

```python
import rustac
import lazycogs
from pyproj import Transformer

# set a target CRS and extent
dst_crs = "EPSG:32615"
dst_bbox = (380000.0, 4928000.0, 420000.0, 4984000.0)

# transform to 4326 for STAC search
transformer = Transformer.from_crs(dst_crs, "epsg:4326", always_xy=True)
bbox_4326 = transformer.transform_bounds(*dst_bbox)

# Search a STAC API and cache results to a local stac-geoparquet file.
await rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=bbox_4326,
)

# Open a fully lazy (band, time, y, x) DataArray. No pixel data is read yet.
# lazycogs does inspect one representative item here: it picks preferred data
# assets, opens one COG per requested band concurrently, and infers a default
# dtype/nodata contract from that representative sample.
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
)
```

### Async loading

When you are already inside an async context (for example, a Jupyter
notebook running on an asyncio loop), you can trigger chunk reads
without blocking the event loop:

```python
# Fetch data asynchronously and load into memory in-place.
subset = await da.isel(x=slice(0, 10), y=slice(0, 10), time=slice(0, 10)).load_async()
```

`load_async` uses xarray's async protocol, which dispatches through
`MultiBandStacBackendArray.async_getitem` and stays on the caller's
event loop. Multiple concurrent chunk reads overlap naturally, so the
async path can be faster than the synchronous `da.compute()` when
reading many chunks inside an already-running loop.

## Custom stores

`lazycogs.open(..., store=...)` accepts any store object that satisfies `async_geotiff.Store`.
For most users, the recommended path is still obstore: leave `store=None` to auto-resolve per-asset stores, or call `lazycogs.store_for()` to build one explicitly.

## Dtype and nodata semantics

See also: [docs guide on dtype and nodata handling](docs/guides/dtype-nodata.md).

When you omit `dtype=`, `lazycogs.open()` samples one representative COG per
requested band and infers one safe output dtype instead of defaulting to
`float32`. That inferred dtype is then enforced at chunk-read time: if a later
asset has a source dtype that cannot be safely represented, compute raises and
asks you to pass `dtype=` explicitly.

When you omit `nodata=`:

- if sampled bands all agree on one scalar nodata sentinel, the returned
  `DataArray` sets `attrs["_FillValue"]`, and masked mosaic output materializes
  with that same sentinel instead of zero
- if sampled bands disagree, `open()` raises `ValueError` and asks you to pass
  `nodata=` explicitly
- if sampled bands have no nodata sentinel, no `_FillValue` metadata is
  attached and `0` remains only an implementation fill value for uncovered
  regions
- if later chunk reads encounter a conflicting source nodata value, compute
  raises and asks you to pass `nodata=` explicitly

Explicit `dtype=` and `nodata=` stay authoritative even when source assets are
heterogeneous.

`lazycogs.open()` also attaches CF/rioxarray-compatible spatial metadata. The
GeoZarr-style `spatial:transform` attribute stays in affine coefficient order,
while `spatial_ref.attrs["GeoTransform"]` is written in GDAL geotransform order
so sliced 2D images and 3D band stacks can be read by rioxarray without repairing
the transform metadata.

Float-only mosaic methods such as `MeanMethod`, `MedianMethod`, and
`StdevMethod` auto-promote inferred integer outputs to `float32`. If you pass
an explicit integer `dtype=` with one of those methods, `open()` raises and
asks for `dtype="float32"` or `dtype="float64"` instead.

### Concurrency notes

- Sync callers submit work to one shared persistent lazycogs event loop.
- CPU-bound reprojection runs on one bounded shared thread pool. Set
  `LAZYCOGS_REPROJECT_WORKERS` before first use to change the default
  `min(os.cpu_count() or 1, 4)` limit. The value is read when the pool is
  first created; changes after that are ignored for the life of the process.
- DuckDB queries yield the event loop by running through one small bounded
  internal submission path and explicit executor instead of on the loop
  thread. On the local benchmark fixture, DuckDB stayed under 2% of
  per-date chunk wall time, so there is no separate per-thread DuckDB
  client pool today.
- `scripts/prepare_benchmark_data.py` also powers offline regression coverage
  under `tests/benchmarks/`, so the contract tests run on local cached files
  instead of live network reads.
- If you need to construct a loop-bound resource for lazycogs internals,
  use `lazycogs.run_on_loop(...)`.
- Low-level callers should use `await lazycogs.read_chunk_async(...)`.

## Documentation

- [Home](https://developmentseed.github.io/lazycogs/) — quickstart and full usage guide
- [Example: Midwest US daily Sentinel-2 array](https://developmentseed.org/lazycogs/notebooks/demo_midwest_daily/)
- [Example: Southwest US monthly low-cloud Sentinel-2 array](https://developmentseed.org/lazycogs/notebooks/demo_southwest_monthly/)
- [Guide: dtype and nodata handling](https://developmentseed.org/lazycogs/guides/dtype-nodata/)
- [Architecture](https://developmentseed.org/lazycogs/architecture/)
- [Contributing](CONTRIBUTING.md)
