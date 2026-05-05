# Spec: rasterix Spatial Indexing for lazycogs DataArrays

## Context

Today lazycogs returns a DataArray with scalar `x` and `y` coordinate arrays (pixel centres in the target CRS) and an internal `dst_affine` stored inside `MultiBandStacBackendArray` that is invisible to xarray operations on the DataArray surface. This means:

- **The affine transform and CRS are not discoverable** through the xarray object. Other spatial xarray tools cannot inspect `da` and know what grid it lives on.
- **Spatial alignment with other raster datasets is not possible.** `xr.align()` operates on indexes, and lazycogs has no spatial index. Two lazycogs arrays on the same grid cannot be aligned, concatenated, or reindexed by xarray without manual coordinate matching.
- **The y-axis coords are eagerly materialized.** For very large output grids (e.g. continental-scale at 10 m resolution) the `y` and `x` coordinate arrays alone can consume significant memory, even though they are a pure function of the affine transform.

[rasterix](https://github.com/xarray-contrib/rasterix) is an `xarray-contrib` project that provides `RasterIndex`, a CRS-aware spatial index built on top of xarray's `CoordinateTransformIndex`. Its capabilities include:

- **Lazy coordinate arrays.** Coordinate labels are computed on demand from the affine transform rather than stored eagerly. For large grids this avoids materializing multi-megabyte coordinate arrays.
- **CRS discoverability.** Because `RasterIndex` inherits from `xproj.ProjIndexMixin`, `da.proj.crs` works out of the box and xarray can check CRS equality during alignment.
- **Spatial alignment.** `join`, `reindex_like`, and `concat` between RasterIndex-backed objects use bounding-box intersection and union computed directly on the index, not on the coordinate values.
- **Efficient slicing.** For rectilinear grids, rasterix uses axis-independent 1-D `AxisAffineTransformIndex` objects. Slicing updates the transform in place rather than slicing through a large coordinate array.

## Goals

- **Primary goal:** Attach a `RasterIndex` to every DataArray returned by `lazycogs.open()`, making the CRS and affine transform discoverable to the xarray ecosystem and enabling alignment with other RasterIndex-backed arrays.
- **Primary goal:** Replace eager `x`/`y` coordinate arrays with lazy coordinate generation derived from the affine transform, reducing memory for large grids.
- **Secondary goal:** Preserve the existing ascending-y coordinate convention so that `da.sel(y=slice(0, 100))` remains intuitive.
- **Secondary goal:** Emit standard spatial metadata attributes so that downstream tools chosen by the user can reconstruct the grid if they choose to persist the array elsewhere.
- **Non-goal:** Adding write APIs to lazycogs (COG writer, Zarr writer, etc.).
- **Non-goal:** Supporting input formats other than COG. rasterix is applied to the *output* DataArray.
- **Non-goal:** Supporting rotated or skewed grids in the initial integration. lazycogs only produces rectilinear north-up grids.

## Constraints & Assumptions

- **lazycogs grids are always rectilinear and north-up.** `compute_output_grid()` produces `Affine(res, 0, minx, 0, -res, maxy)`.
- **y-coordinates in lazycogs are ascending (south to north).** This is an intentional UX decision. `da.sel(y=slice(0, 100))` selects the southernmost 10 km of a UTM grid. This must be preserved after adopting rasterix.
- **rasterix `RasterIndex.from_transform` generates coordinates from the affine transform.** For a standard top-down affine (negative y-scale), the derived y coordinates are descending (north to south), which conflicts with lazycogs' ascending convention. **This is the main integration challenge.** See Open Questions for discussion.
- **xproj is required for CRS-aware indexing.** Adding rasterix implies adding `xproj` as a dependency.
- **Chunking must preserve the index.** When `chunks=...` is passed, xarray calls `.chunk()` on the DataArray. For rectilinear grids, rasterix's `isel()` returns a new `RasterIndex` with an updated transform on slice indexing. This needs verification with `LazilyIndexedArray` → dask transition.
- **Backward compatibility.** Existing code that accesses `da.coords['x']` or iterates over `da['y'].values` must continue to work. The coordinate variable values themselves must remain identical.

## What rasterix adds (and what it does not)

### `.sel()` already works today

Because lazycogs attaches explicit 1-D `x` and `y` coordinate arrays in projected units, users can already perform label-based selection:

```python
da.sel(x=slice(-400_000, -200_000), y=slice(2_500_000, 2_700_000))
```

This works because xarray's default `PandasIndex` maps coordinate labels to integer positions. rasterix is **not** needed for this. What rasterix adds is:

- **Position-based selection through the transform.** The index maps real-world labels to pixel positions using the affine transform directly, rather than searching through a coordinate array. This is faster for very large coordinate arrays.
- **Spatial alignment between arrays.** `xr.align(da1, da2)` can intersect or union their bounding boxes because the index knows the transform.
- **Lazy coordinates.** Coordinate labels are generated on demand from the transform rather than stored.

## Architecture Overview

The change is concentrated in the Phase-0 `open()` path (`_core.py` / `_grid.py`). Phase-1 chunk reading is untouched.

```
open()
  ├── compute_output_grid()  ──► (dst_affine, width, height, x_coords, y_coords)
  │                                [unchanged coordinate math]
  │
  ├── _build_dataarray()
  │      ├── Create MultiBandStacBackendArray [unchanged]
  │      ├── Create xr.Variable [unchanged]
  │      ├── .chunk(chunks) if requested [unchanged]
  │      ├── Build coords dict [unchanged]
  │      ├── NEW: RasterIndex.from_transform(dst_affine, width, height, crs=dst_crs)
  │      ├── NEW: xarray Coordinates.from_xindex(index) → replaces default x/y indexes
  │      └── Build DataArray with new coords
  │
  └── Return DataArray with RasterIndex on x and y
```

### What changes in the DataArray

**Before (today)**

```python
>>> da = lazycogs.open(...)
>>> da.xindexes
{'band': PandasIndex, 'time': PandasIndex, 'y': PandasIndex, 'x': PandasIndex}
>>> da.attrs
{'_stac_backend': <MultiBandStacBackendArray ...>,
 '_stac_time_coords': ...}
```

**After**

```python
>>> da = lazycogs.open(...)
>>> da.xindexes
{'band': PandasIndex, 'time': PandasIndex,
 'y': AxisAffineTransformIndex(...), 'x': AxisAffineTransformIndex(...)}
>>> da.proj.crs
<Projected CRS: EPSG:5070>
>>> da.xindexes['x'].transform()   # rasterix center_transform
Affine(10.0, 0.0, -399995.0, 0.0, -10.0, 2700005.0)
>>> da.attrs
{'_stac_backend': <MultiBandStacBackendArray ...>,
 '_stac_time_coords': ...,
 'grid_mapping': 'spatial_ref',
 'zarr_conventions': [{'name': 'spatial:', ...}, {'name': 'proj:', ...}],
 'spatial:transform': [10.0, 0.0, -400000.0, 0.0, -10.0, 2700000.0],
 'proj:code': 'EPSG:5070',
 ...}
```

## API / Interface Design

No new public API. The `RasterIndex` is attached automatically inside `open()`. There is no `spatial_index` toggle. If a user encounters an unexpected rasterix/xproj bug, the escape hatch is to pin an older version of lazycogs.

### New internal helper

```python
# lazycogs/_spatial_index.py  (new module)

from __future__ import annotations

from typing import TYPE_CHECKING

from affine import Affine
from pyproj import CRS

if TYPE_CHECKING:
    import xarray as xr


def attach_raster_index(
    da: xr.DataArray,
    *,
    dst_affine: Affine,
    dst_crs: CRS,
) -> xr.DataArray:
    """Attach a RasterIndex to the x/y dimensions of a lazycogs DataArray.

    Parameters
    ----------
    da :
        DataArray returned by _build_dataarray (before attrs are populated).
    dst_affine :
        Top-left-corner affine transform of the full output grid (GDAL
        convention).  This is the same transform stored in
        MultiBandStacBackendArray.dst_affine.
    dst_crs :
        Target CRS.

    Returns
    -------
    DataArray with RasterIndex on x and y, plus spatial metadata attrs.
    """
    ...
```

## Data Model

### RasterIndex construction parameters

| Parameter | Value | Notes |
|---|---|---|
| `affine` | `dst_affine` from `compute_output_grid()` | Top-left origin. rasterix shifts internally by +0.5 for pixel-centre math. |
| `width` | `dst_width` | From `compute_output_grid()`. |
| `height` | `dst_height` | From `compute_output_grid()`. |
| `x_dim` | `"x"` | Matches existing lazycogs dimension name. |
| `y_dim` | `"y"` | Matches existing lazycogs dimension name. |
| `crs` | `dst_crs` | ``pyproj.CRS`` object passed to ``open()``. |

### Metadata attributes

The spec includes metadata attributes that describe the spatial grid. These are emitted so that downstream tools chosen by the user can reconstruct the grid. They are not acted on by lazycogs itself.

1. **CF / GDAL convention**
   - `grid_mapping: "spatial_ref"`
   - A `"spatial_ref"` coordinate containing the CRS WKT and `GeoTransform` attribute.

2. **Zarr Spatial Convention**
   - `'zarr_conventions': [{"name": "spatial:", "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"}, ...]`
   - `'spatial:transform': [a, b, c, d, e, f]` (the top-left-corner affine, not pixel-centre)
   - `'spatial:transform_type': 'affine'`
   - `'spatial:registration': 'pixel'`

3. **Zarr geo-proj convention**
   - `'zarr_conventions'` list also contains `{"name": "proj:", "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f"}`
   - `'proj:code': 'EPSG:5070'` (or `proj:wkt2`, `proj:projjson` as fallback)

## Integration Points

### rasterix / xproj

- `rasterix.RasterIndex.from_transform` creates the index.
- `xarray.Coordinates.from_xindex(index)` builds the coordinate bundle.
- `xproj` provides the `da.proj.crs` accessor and CRS equality checks during alignment.

### lazycogs internal modules

| Module | Change |
|---|---|
| `_core.py` | After ``xr.DataArray(...)`` is built, call `attach_raster_index()` before returning. |
| `_grid.py` | No change. `compute_output_grid()` already produces everything `attach_raster_index` needs. |
| `_backend.py` | No change. `MultiBandStacBackendArray` retains the raw top-left affine for chunk math. |
| `_explain.py` | No change. The accessor reads `_stac_backend` from `attrs`. |
| `_spatial_index.py` | **New module.** Encapsulates rasterix index creation and metadata emission. Keeps rasterix imports localized. |

### Interoperability with other spatial xarray tools

| Tool | Benefit |
|---|---|
| `odc-geo` / `odc-stac` | Can derive a `Geobox` directly from the `RasterIndex` because transform and CRS are explicit xarray indexes. |
| `stackstac` | Can align or concat stackstac output with lazycogs output when both carry matching `RasterIndex` instances. |
| Any rasterix-aware reader | Can open lazycogs-written Zarr data and reconstruct the full spatial grid from the embedded conventions. |

## Migration Path

This is an additive change with no deprecation cycle:

- The coordinate variables (`x`, `y`) remain identical in values and shape.
- The public API of `lazycogs.open()` gains no new arguments.
- Existing code that accesses `da.coords['x']` or iterates over `da['y'].values` continues to work unchanged.

Because there is no `spatial_index` toggle, the integration must be rock-solid before release. The recommended path is to ship behind a feature flag during development (e.g. an environment variable) and remove the flag once tests pass.

## Testing Strategy

### Unit tests

1. **Index presence.** `open(...)` returns a DataArray whose `xindexes['x']` is an `AxisAffineTransformIndex` (wrapped by `RasterIndex`).
2. **CRS attachment.** `da.proj.crs` equals the `crs=` argument passed to `open()`.
3. **Transform round-trip.** `da.xindexes['x'].transform()` reconstructs the expected pixel-centre affine.
4. **Metadata attributes.** `da.attrs` contains `grid_mapping` and a `spatial_ref` coordinate after attachment.
5. **Zarr convention attrs.** When opened with `crs="EPSG:5070"`, the DataArray carries `spatial:transform`, `proj:code`, and the correct `zarr_conventions` list.
6. **Ascending y preserved.** `da.coords['y'].values` is strictly increasing (bottom to top) after rasterix attachment.

### Integration tests

1. **Slice preserves index.** `da.isel(x=slice(10, 20))` yields a DataArray whose `RasterIndex` has an updated transform and smaller bbox.
2. **Chunking preserves index.** `open(..., chunks={"x": 512, "y": 512})` produces a dask-backed DataArray that still carries `RasterIndex` on chunk descriptors.
3. **Alignment between two lazycogs arrays.** Create two arrays with adjacent bboxes and the same CRS/resolution; `xr.align(da1, da2, join="outer")` succeeds and produces matching transforms.

## Decision Log

| Decision | Options Considered | Rationale |
|---|---|---|
| Attach RasterIndex automatically, no toggle | Automatic only; or `spatial_index=True/False` flag. | No toggle keeps the API surface minimal. If bugs surface, users pin an older version. |
| Use `RasterIndex.from_transform` directly | `assign_index` (guesses from attrs) vs. explicit construction. | `from_transform` is correct because lazycogs already knows the exact transform, width, height, and CRS. |
| Emit all three metadata conventions | Only CF/GDAL; only Zarr; both. | Both. The attrs are tiny and maximize interoperability with any tool the user chooses. |
| Make rasterix required | Optional with import fallback vs. required. | Required. rasterix + xproj are lightweight pure Python. An optional path adds import complexity for marginal benefit. |
| Pass top-left or centre transform to rasterix | Top-left (GDAL) vs. centre. | `from_transform` docs say "Should represent pixel top-left corners" and internally applies `Affine.translation(0.5, 0.5)`. We pass the same top-left transform that lazycogs computes. |
| Leave write support out of scope | Include COG/GeoZarr/netCDF write discussion vs. exclude entirely. | The user explicitly wants write pathways out of scope. The spec focuses on indexing, alignment, lazy coords, and metadata portability. |

## Open Questions

1. **Ascending y coordinates vs. rasterix's descending transform-derived coords.** This is the main blocking question.

   rasterix `RasterIndex.from_transform` with a standard top-down affine (`e < 0`) generates y coordinates via `AxisAffineTransform.forward()`: position 0 maps to the top (largest y), position height-1 maps to the bottom (smallest y). The resulting coordinate array is descending.

   lazycogs deliberately makes `y` ascending (smallest y at position 0, largest y at position height-1) so that `da.sel(y=slice(0, 100))` selects the southern end of the grid.

   Potential approaches:
   - **Option A:** Pass rasterix a synthetic y-axis transform with positive y-scale and adjusted origin, then override the coord variable values after index creation. This may break `sel()` because the index's reverse mapping would not match the coord labels.
   - **Option B:** Skip `Coordinates.from_xindex()` entirely. Manually construct `AxisAffineTransformIndex` objects with custom transforms that produce ascending y coords, attach them via `da.set_xindex()`, and supply our own coord variables. This bypasses rasterix's normal path but gives full control.
   - **Option C:** Patch rasterix to support an `ascending_y` flag or to respect externally-supplied coordinate arrays instead of generating them from the transform.
   - **Option D:** Accept descending y coords from rasterix and flip the lazycogs data array convention to match standard raster orientation (top-down y). This is a breaking change to the user-facing API.

   **Recommendation:** investigate Option B first. It gives us full control over both the index and the coordinate values without requiring upstream rasterix changes. If Option B proves too complex, Option C (a small rasterix patch) is the next best path.

2. **Dask chunking with CoordinateTransformIndex.** When `da.chunk(chunks)` is called, xarray creates new indexes for each chunk via `index.isel()`. For rectilinear grids, rasterix's `isel()` returns a new `RasterIndex` with an updated transform. This needs end-to-end verification with the `LazilyIndexedArray` → dask transition path that lazycogs uses.

3. **Zarr convention registration format.** Confirm the exact structure the Zarr Spatial Convention v1.0 requires for `zarr_conventions` and verify it matches what rasterix reads when opening Zarr data.

4. **Performance impact at `open()` time.** rasterix index creation is pure Python + affine math; it should be negligible compared to DuckDB queries. We should benchmark `open()` before and after to confirm.

5. **`_stac_backend` attr serialization.** `MultiBandStacBackendArray` is stored in `da.attrs['_stac_backend']`. xarray's `to_zarr` and `to_netcdf` will attempt to serialize this. We may need to move it out of `attrs` into accessor state so that only spatial metadata ends up in written files.

## Status

- [x] Designing
- [ ] Approved — ready to plan
- [ ] Implementing
- [ ] Implemented

## References

- [rasterix repository](https://github.com/xarray-contrib/rasterix)
- [rasterix docs — Creating RasterIndex](https://rasterix.readthedocs.io/en/latest/raster_index/creating.html)
- [rasterix docs — Alignment](https://rasterix.readthedocs.io/en/latest/raster_index/aligning.html)
- [Zarr Spatial Convention v1.0](https://zarr-specs.readthedocs.io/en/latest/v3/conventions/spatial/v1.0.html)
- [Zarr geo-proj convention](https://github.com/zarr-conventions/geo-proj)
- [xarray Flexible Indexes design docs](https://docs.xarray.dev/en/stable/internals/how-to-add-new-index.html)
- lazycogs `_core.py`, `_grid.py`, `_backend.py` (current source)
