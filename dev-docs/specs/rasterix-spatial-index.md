# Spec: rasterix Spatial Indexing for lazycogs DataArrays

## Context

Today lazycogs returns a DataArray with lazy `x` and `y` coordinates backed by a `RasterIndex` (derived from the affine transform) and an internal `dst_affine` stored inside `MultiBandStacBackendArray` that is invisible to xarray operations on the DataArray surface.

[rasterix](https://github.com/xarray-contrib/rasterix) is an `xarray-contrib` project that provides `RasterIndex`, a CRS-aware spatial index built on top of xarray's `CoordinateTransformIndex`. Its capabilities include:

- **Lazy coordinate arrays.** Coordinate labels are computed on demand from the affine transform rather than stored eagerly. For large grids this avoids materializing multi-megabyte coordinate arrays.
- **CRS discoverability.** Because `RasterIndex` inherits from `xproj.ProjIndexMixin`, `da.proj.crs` works out of the box and xarray can check CRS equality during alignment.
- **Spatial alignment.** `join`, `reindex_like`, and `concat` between RasterIndex-backed objects use bounding-box intersection and union computed directly on the index, not on the coordinate values.
- **Efficient slicing.** For rectilinear grids, rasterix uses axis-independent 1-D `AxisAffineTransformIndex` objects. Slicing updates the transform in place rather than slicing through a large coordinate array.

## Goals

- **Primary goal:** Attach a `RasterIndex` to every DataArray returned by `lazycogs.open()`, making the CRS and affine transform discoverable to the xarray ecosystem and enabling alignment with other RasterIndex-backed arrays.
- **Primary goal:** Replace eager `x`/`y` coordinate arrays with lazy coordinate generation derived from the affine transform, reducing memory for large grids.
- **Secondary goal:** Emit standard spatial metadata attributes so that downstream tools chosen by the user can reconstruct the grid if they choose to persist the array elsewhere.
- **Non-goal:** Adding write APIs to lazycogs (COG writer, Zarr writer, etc.).
- **Non-goal:** Supporting input formats other than COG. rasterix is applied to the *output* DataArray.
- **Non-goal:** Supporting rotated or skewed grids in the initial integration. lazycogs only produces rectilinear north-up grids.

## Constraints & Assumptions

- **lazycogs grids are always rectilinear and north-up.** `compute_output_grid()` produces `Affine(res, 0, minx, 0, -res, maxy)`.
- **y-coordinates follow the standard raster convention (descending, north to south).** lazycogs previously flipped y-coordinates to be ascending, but this deviated from the affine transform and from conventions used by `odc-stac`, `rioxarray`, and other spatial xarray tools. After adopting rasterix, y is descending, consistent with the top-down affine (`e < 0`). `da.sel(y=slice(2_700_000, 2_500_000))` selects the southernmost portion of a UTM grid.
- **rasterix `RasterIndex.from_transform` generates coordinates from the affine transform.** For a standard top-down affine (negative y-scale), the derived y coordinates are descending (north to south), which matches lazycogs' updated convention.
- **xproj is required for CRS-aware indexing.** Adding rasterix implies adding `xproj` as a dependency.
- **Chunking preserves the index.** When `chunks=...` is passed, xarray calls `.chunk()` on the DataArray. For rectilinear grids, rasterix's `isel()` returns a new `RasterIndex` with an updated transform on slice indexing. This has been verified with the `LazilyIndexedArray` → dask transition path.

## What rasterix adds (and what it does not)

### `.sel()` already works today

Because lazycogs attaches explicit 1-D `x` and `y` coordinate arrays in projected units, users can already perform label-based selection:

```python
da.sel(x=slice(-400_000, -200_000), y=slice(2_700_000, 2_500_000))
```

This works because xarray's default `PandasIndex` maps coordinate labels to integer positions. rasterix is **not** needed for this. What rasterix adds is:

- **Position-based selection through the transform.** The index maps real-world labels to pixel positions using the affine transform directly, rather than searching through a coordinate array. This is faster for very large coordinate arrays.
- **Spatial alignment between arrays.** `xr.align(da1, da2)` can intersect or union their bounding boxes because the index knows the transform.
- **Lazy coordinates.** Coordinate labels are generated on demand from the transform rather than stored.

## Architecture Overview

The change is concentrated in the Phase-0 `open()` path (`_core.py` / `_grid.py`). Phase-1 chunk reading is untouched. `compute_output_grid()` returns only the affine transform and dimensions — no eager coordinate arrays.

```
open()
  ├── compute_output_grid()  ──► (dst_affine, width, height)
  │                                [no eager x/y coordinate arrays]
  │
  ├── _build_dataarray()
  │      ├── Create MultiBandStacBackendArray [unchanged]
  │      ├── Create xr.Variable [unchanged]
  │      ├── .chunk(chunks) if requested [unchanged]
  │      ├── Build RasterIndex.from_transform(dst_affine, width, height, crs=dst_crs)
  │      ├── xarray Coordinates.from_xindex(index) → lazy x/y coords
  │      ├── Add spatial_ref coord, grid_mapping + Zarr convention attrs
  │      └── Build DataArray with band/time coords + spatial index coords
  │
  └── Return DataArray with RasterIndex on x and y
```

### What changes in the DataArray

**Before**

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
 'spatial:transform_type': 'affine',
 'spatial:registration': 'pixel',
 'proj:code': 'EPSG:5070',
 ...}
```

## API / Interface Design

No new public API. The `RasterIndex` is attached automatically inside `open()`. There is no `spatial_index` toggle. If a user encounters an unexpected rasterix/xproj bug, the escape hatch is to pin an older version of lazycogs.

### Internal implementation (inlined in `_build_dataarray`)

Rather than a separate module, `RasterIndex` creation and spatial metadata emission live directly inside `_core.py::_build_dataarray()`. This avoids the overhead of allocating temporary `x_coords`/`y_coords` numpy arrays only to replace them with lazy coordinates.

```python
# Inside _core.py::_build_dataarray()

index = RasterIndex.from_transform(
    dst_affine,
    width=dst_width,
    height=dst_height,
    x_dim="x",
    y_dim="y",
    crs=dst_crs,
)
spatial_coords = Coordinates.from_xindex(index)

# CF / GDAL spatial_ref coordinate
gt = [dst_affine.a, dst_affine.b, dst_affine.c,
      dst_affine.d, dst_affine.e, dst_affine.f]
spatial_ref = DataArray(
    np.array(0),
    attrs={
        "crs_wkt": dst_crs.to_wkt(),
        "GeoTransform": " ".join(str(v) for v in gt),
    },
)
attributes = {
    "grid_mapping": "spatial_ref",
    "zarr_conventions": [...],
    "spatial:transform": gt,
    "spatial:transform_type": "affine",
    "spatial:registration": "pixel",
    # ... plus _stac_backend, _stac_time_coords, proj:code, etc.
}
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

The DataArray carries metadata attributes that describe the spatial grid. These are emitted so that downstream tools chosen by the user can reconstruct the grid if they choose to persist the array elsewhere. They are not acted on by lazycogs itself.

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
   - `'proj:code': 'EPSG:5070'` (or `proj:wkt2` as fallback)

## Integration Points

### rasterix / xproj

- `rasterix.RasterIndex.from_transform` creates the index.
- `xarray.Coordinates.from_xindex(index)` builds the coordinate bundle.
- `xproj` provides the `da.proj.crs` accessor and CRS equality checks during alignment.

### lazycogs internal modules

| Module | Change |
|---|---|
| `_core.py` | `_build_dataarray()` builds `RasterIndex.from_transform(...)` inline, replaces explicit x/y coords with `Coordinates.from_xindex(...)`, and adds spatial metadata attrs. |
| `_grid.py` | Simplified: returns only `(dst_affine, dst_width, dst_height)` — no eager x/y coordinate arrays. |
| `_backend.py` | No change. `MultiBandStacBackendArray` retains the raw top-left affine for chunk math. |
| `_explain.py` | No change. The accessor reads `_stac_backend` from `attrs`. |

### Interoperability with other spatial xarray tools

| Tool | Benefit |
|---|---|
| `odc-geo` / `odc-stac` | Can derive a `Geobox` directly from the `RasterIndex` because transform and CRS are explicit xarray indexes. |
| `stackstac` | Can align or concat stackstac output with lazycogs output when both carry matching `RasterIndex` instances. |
| Any rasterix-aware reader | Can open lazycogs-written Zarr data and reconstruct the full spatial grid from the embedded conventions. |

## Migration Path

This release contains a breaking change to y-coordinate ordering:

- **y-coordinate values are reversed.** The `y` coordinate array is descending (north to south) instead of ascending. Code that slices with `y=slice(min_y, max_y)` must be updated to `y=slice(max_y, min_y)`.
- The public API of `lazycogs.open()` gains no new arguments.
- Existing code that accesses `da.coords['x']` or iterates over `da['y'].values` will still execute, but the order of `y` values is inverted.

## Testing Strategy

### Unit tests

1. **Index presence.** `open(...)` returns a DataArray whose `xindexes['x']` is an `AxisAffineTransformIndex` (wrapped by `RasterIndex`).
2. **CRS attachment.** `da.proj.crs` equals the `crs=` argument passed to `open()`.
3. **Transform round-trip.** `da.xindexes['x'].transform()` reconstructs the expected pixel-centre affine.
4. **Metadata attributes.** `da.attrs` contains `grid_mapping` and a `spatial_ref` coordinate after attachment.
5. **Zarr convention attrs.** When opened with `crs="EPSG:5070"`, the DataArray carries `spatial:transform`, `proj:code`, and the correct `zarr_conventions` list.
6. **Descending y (standard raster orientation).** `da.coords['y'].values` is strictly decreasing (north to south) after rasterix attachment, matching the top-down affine transform.

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
| Adopt descending y coordinates (north to south) | Keep ascending y (south to north) via manual flip vs. accept rasterix's descending coords. | The broader ecosystem (`odc-stac`, `rioxarray`, GDAL) uses descending y consistent with top-down affine transforms. Staying true to the transform removes integration friction, even though it breaks existing lazycogs slicing assumptions. |
| Inline RasterIndex creation in `_core.py` | Separate `_spatial_index.py` module vs. inline in `_build_dataarray`. | Inlining avoids a level of indirection and eliminates the need to allocate temporary numpy coordinate arrays that would immediately be discarded. It also keeps the rasterix import surface in a single file. |
| Simplify `compute_output_grid()` | Return x/y arrays + keep grid stable vs. return only affine + dims. | Dropping coordinate arrays from the return value avoids wasted allocation. The coordinates are now generated lazily by the `RasterIndex`. |

## Open Questions

1. **Resolved: y-axis coordinate ordering.** After surveying `odc-stac`, `rioxarray`, and other spatial-xarray tools, the prevailing convention is a descending y-axis that remains consistent with the affine transform (`e < 0`). Manually flipping coordinates to ascending was an outlier that complicated integration.

   **Decision:** Adopt standard top-down raster orientation. The `y` coordinate array is descending (north to south). This is a breaking change for code that relied on the previous ascending-y convention, but it makes lazycogs interoperable with rasterix, `odc-geo`, and similar libraries without workarounds.

2. **Resolved: Dask chunking with CoordinateTransformIndex.** Verified via tests: when `da.chunk(chunks)` is called, xarray creates new indexes for each chunk via `index.isel()`. For rectilinear grids, rasterix's `isel()` returns a new `RasterIndex` with an updated transform. The `LazilyIndexedArray` → dask transition path works correctly.

3. **Resolved: Zarr convention registration format.** The `zarr_conventions` list follows the documented UUID format for the Spatial Convention v1.0 and geo-proj convention.

4. **Resolved: Performance impact at `open()` time.** The rasterix index creation is pure Python + affine math. It adds negligible overhead compared to DuckDB queries.

5. **Resolved: No separate `_spatial_index.py` module.** The RasterIndex creation and spatial metadata setup are inlined directly in `_core.py::_build_dataarray()`. This eliminates the overhead of allocating temporary coordinate arrays and avoids an unnecessary module boundary.

## Status

- [x] Designing
- [x] Approved — ready to plan
- [x] Implementing
- [x] Implemented

## References

- [rasterix repository](https://github.com/xarray-contrib/rasterix)
- [rasterix docs — Creating RasterIndex](https://rasterix.readthedocs.io/en/latest/raster_index/creating.html)
- [rasterix docs — Alignment](https://rasterix.readthedocs.io/en/latest/raster_index/aligning.html)
- [Zarr Spatial Convention v1.0](https://zarr-specs.readthedocs.io/en/latest/v3/conventions/spatial/v1.0.html)
- [Zarr geo-proj convention](https://github.com/zarr-conventions/geo-proj)
- [xarray Flexible Indexes design docs](https://docs.xarray.dev/en/stable/internals/how-to-add-new-index.html)
- lazycogs `_core.py`, `_grid.py`, `_backend.py` (current source)
