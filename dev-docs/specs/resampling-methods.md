# Spec: Raster Resampling Methods in lazycogs

## Context

lazycogs currently reprojects source COG pixels to the output grid using a pure-numpy nearest-neighbor warp map in `_reproject.py`. The `compute_warp_map` / `apply_warp_map` pair stores source column and row indices for every destination pixel, then samples directly. This works well for many workflows but is the only resampling strategy available.

Users working with spectral data, DEMs, or any continuous-field imagery will want higher-quality interpolation: bilinear, bicubic, Lanczos, and so on. Supporting these methods is not a drop-in replacement because:

- Non-point-sampling kernels are moving-window operations that require reading a "halo" of extra source pixels around the geometrically-mapped window.
- That halo changes **asset discovery** (the DuckDB spatial query). An item whose footprint is just outside the strict chunk bbox may still have edge pixels needed for interpolation, so the STAC search bbox must be expanded to include those items.
- The choice interacts with COG overview selection: an overview may already have been downsampled with a method (often nearest-neighbor) that conflicts with the user's requested reprojection resampling.
- Performance characteristics change: more source data must be fetched, and the CPU work per pixel increases.

This spec defines how lazycogs should expose resampling options while keeping the existing no-GDAL, numpy-first architecture.

## Goals

- **Primary goal:** Allow callers of `lazycogs.open()` to request a reprojection resampling method other than nearest-neighbor (e.g. bilinear, bicubic).
- **Secondary goal:** Automatically compute and request the correct source-pixel halo for each kernel so that edge pixels are interpolated correctly.
- **Secondary goal:** Document the overview/resampling interaction so users understand quality trade-offs.
- **Non-goal:** Controlling the resampling algorithm used to *generate* COG overviews. That is set at production time by the data provider.
- **Non-goal:** Supporting every GDAL resampling mode. The initial set is: `nearest`, `bilinear`, and `cubic`.
- **Non-goal:** Per-band resampling control. A single parameter applies to all bands in the opened array.

## Constraints & Assumptions

- **No GDAL / no rasterio dependency.** All resampling logic must be pure numpy or standard-library Python.
- **`interpn` is the interpolation backend** for non-nearest resampling. It provides fast, allocation-free multilinear and multicubic interpolation via Rust/Python bindings.
- **Backward compatibility:** the default resampling remains `nearest`; existing code must not change behavior.
- **async-geotiff `read(window=...)` is the bottleneck.** Extra halo pixels are requested by expanding the `Window` passed to `async_geotiff`; the library itself does not need to know about resampling.
- **Mosaic methods are applied after reprojection.** Resampling operates on individual source tiles before they are fed into the `MosaicMethodBase` accumulator.
- **Chunk reads are independent.** A requested output chunk may map to disjoint source tiles; each tile is read and reprojected separately before mosaicking.

## Architecture Overview

The change touches three layers:

1. **Public API** — `lazycogs.open()` gains a `resampling` parameter.
2. **Chunk planning** — `_chunk_reader.py` must expand source `Window`s by a kernel-dependent halo before reading.
3. **Reprojection** — `_reproject.py` must grow from a single nearest-neighbor path to a pluggable kernel system that can sample with interpolation.

```
open(resampling="bilinear")
       │
       ▼
MultiBandStacBackendArray ----► _ChunkReadPlan (carries ResamplingKernel)
       │
       ├──► _resolve_spatial_window()   (bbox_4326 expanded by halo distance)
       │         │
       │         └──► _search_items()   (DuckDB bbox includes adjacent items)
       │
       ▼
async_mosaic_chunk() ----► _read_item_band()
       │                       │
       │                       ├──► GeoTIFF.open()
       │                       ├──► _select_overview()   (unchanged logic)
       │                       ├──► _native_window()     (expanded by halo)
       │                       └──► reader.read()        (reads enlarged window)
       │
       ▼
_apply_bands_with_warp_cache()
       │
       ├──► nearest path ──► apply_warp_map()    (indexing)
       └──► interp path ──► apply_interp_map()  (interpn)
```

### Kernel abstraction

A `ResamplingKernel` dataclass describes:

- `name`: string identifier, validated at `open()` time.
- `radius`: how many pixels beyond the strict geometric bounds must be fetched (e.g. bilinear = 1, cubic = 2).
- `interpn_class`: the `interpn` class used to evaluate the kernel (e.g. `interpn.MultilinearRegular`).

Kernels are singleton instances looked up by name. The radius is used in `_native_window` to expand the read window; `interpn_class` is used in `_apply_bands_with_warp_cache` to produce the destination chunk.

### Bbox expansion for asset discovery

Before any COG is opened, `_resolve_spatial_window` in `_backend.py` computes the chunk bbox in EPSG:4326 for the DuckDB search. With nearest-neighbor, any item that does not overlap this bbox cannot contribute pixels. With interpolation, an item whose footprint is adjacent may have edge pixels needed for the halo.

The search bbox is therefore expanded by a ground distance computed from the kernel radius and the chunk resolution:

```python
# In _resolve_spatial_window or its caller
halo_distance = kernel.radius * abs(chunk_affine.a)   # dst CRS units
expanded_bbox_dst = (
    minx - halo_distance,
    miny - halo_distance,
    maxx + halo_distance,
    maxy + halo_distance,
)
```

`expanded_bbox_dst` is then transformed to EPSG:4326 to produce `chunk_bbox_4326`. This is a conservative expansion: if a source is finer-resolution than the chunk, its halo covers less ground distance, so the expanded bbox is certainly large enough. If a source is coarser-resolution (upsampling case), the true halo distance is larger; this is documented as a known limitation, and users can increase `halo_distance` manually via `halo_pixels=` if necessary.

### Halo expansion at read time

`_native_window` computes the smallest integer pixel window that fully covers the projected chunk bounding box. For interpolation, this window is expanded by the kernel's `radius` in all four directions, then clipped to the image extent:

```python
# current
window = Window(col_min, row_min, col_max - col_min, row_max - row_min)

# with halo
halo = kernel.radius
window = Window(
    max(0, col_min - halo),
    max(0, row_min - halo),
    min(width, col_max + halo) - max(0, col_min - halo),
    min(height, row_max + halo) - max(0, row_min - halo),
)
```

Because the source window is larger, the warp/interp map must account for the halo offset: source pixel coordinates are shifted by the halo so that the interpolator receives the correct continuous coordinates relative to the enlarged array.

### Coordinate mapping for interpolation

Nearest-neighbor uses integer source indices. Interpolating kernels need fractional source coordinates (continuous pixel centers). The current `WarpMap` stores integer indices. We introduce an `InterpMap` dataclass that stores fractional column and row coordinates:

```python
@dataclass
class InterpMap:
    src_col_frac: np.ndarray   # shape (dst_h, dst_w), float64
    src_row_frac: np.ndarray   # shape (dst_h, dst_w), float64
    valid: np.ndarray          # bool, True where the source contains the sample
```

`compute_interp_map` is identical to `compute_warp_map` except it does **not** floor the fractional coordinates. `valid` is derived from whether the fractional coordinate, plus the kernel radius, lies inside the enlarged source array bounds.

Two implementation paths exist for the actual interpolation:

**Path A: `interpn` (default for non-nearest methods)**
For each source tile, lazycogs builds the enlarged source array, constructs an `interpn` interpolator (`MultilinearRegular` for bilinear, `MulticubicRegular` for cubic), and evaluates it at the fractional source coordinates computed by `compute_interp_map`. Because `interpn` operates on flattened arrays with regular-grid metadata, the source array is flattened in C-order and coordinate arrays are offset by the halo window's origin.

`interpn` performs no heap allocations during evaluation and is significantly faster than pure-numpy alternatives. Boundary handling is done outside `interpn` by masking out-of-bounds destination pixels with `nodata`.

**Path B: nearest-neighbor fast path**
Keep the existing `compute_warp_map` + `apply_warp_map` integer-indexing path for `nearest`. This avoids any interpolation overhead and preserves exact backward compatibility.

### Overview interaction

`_select_overview` chooses the coarsest overview whose pixel size is `<= target_res`. This logic does not change.

However, there is a quality interaction: if the user asks for `resampling="cubic"` but the selected overview was generated with nearest-neighbor, the output is effectively nearest-then-cubic. This is usually acceptable because:

- The overview is already at or finer than the requested resolution, so the NN overview is just a pre-filtering step.
- For downsampling by large integer factors (e.g. 30 m native to 120 m output), using the 120 m overview avoids reading 16x the data. The slight quality loss from the provider's NN overview is often outweighed by the I/O savings.

For users who want full quality, the existing `max_overview_level` parameter can be set to `0` to disable overviews entirely (forcing full-resolution reads). The spec proposes adding an `overview_level: int | None = None` parameter to `open()` so users can pin a specific level independently.

**Addressing the conflict question:** when a user requests `resampling="cubic"` at 120 m from 30 m data, and a nearest-neighbor 120 m overview exists, the spec recommends:

1. Default behavior: use the overview (fast), because the user's intent is "give me 120 m output with cubic resampling" and the overview satisfies the resolution constraint.
2. If the user sets `max_overview_level=0` (disabling overviews), the native 30 m data is read and cubic resampling applied directly. This gives the theoretically best quality at the cost of reading 16x more pixels.
3. If the user passes `overview_level=0`, use the finest overview (not native), and so on for higher levels.

This keeps the API simple while giving power users an escape hatch.

## API / Interface Design

### `lazycogs.open()`

```python
def open(
    href: str,
    *,
    ...,
    resampling: str = "nearest",
    overview_level: int | None = None,
    ...,
) -> xr.DataArray:
    ...
```

- `resampling`: one of `"nearest"`, `"bilinear"`, `"cubic"`. Raises `ValueError` for unknown strings.
- `halo_pixels`: optional override for the number of pixels to expand the search/read bbox in each direction. When `None` (default), the value is taken from the kernel's `radius`.
- `overview_level`: one of `None` or a non-negative integer. Raises `ValueError` for invalid values.
  - `None` (default): automatic overview selection via `_select_overview`.
  - `0`, `1`, ...: pin to a specific overview level index (`0` = finest overview).

For backward compatibility, the default `resampling="nearest"` continues to use the existing fast path with integer `WarpMap` indexing.

### Internal dataclass

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ResamplingKernel:
    name: str
    radius: int   # pixel halo on each side
    interpn_class: type | None   # interpn interpolator class; None == nearest
```

```python
# registry
_KERNELS: dict[str, ResamplingKernel] = {
    "nearest":  ResamplingKernel("nearest",  0, None),
    "bilinear": ResamplingKernel("bilinear", 1, interpn.MultilinearRegular),
    "cubic":    ResamplingKernel("cubic",    2, interpn.MulticubicRegular),
}
```

### Private function signatures

```python
# _reproject.py

def compute_interp_map(
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
) -> InterpMap:
    """Build fractional pixel-coordinate mapping for interpolation."""
    ...

def apply_interp_map(
    data: np.ndarray,
    interp_map: InterpMap,
    kernel: ResamplingKernel,
    nodata: float | None = None,
) -> np.ndarray:
    """Resample source array using fractional coordinates and an interpn kernel.

    The source ``data`` is assumed to be the enlarged read window (including halo).
    Coordinates in ``interp_map`` are shifted by the window origin before evaluation
    so that they are relative to the sub-array passed to ``interpn``.
    """
    ...
```

```python
# _chunk_reader.py

def _native_window(
    geotiff: GeoTIFF | Overview,
    bbox_native: tuple[float, float, float, float],
    width: int,
    height: int,
    halo: int = 0,
) -> Window | None:
    ...
```

## Data Model

### `InterpMap`

```python
@dataclass
class InterpMap:
    src_col_frac: np.ndarray   # (dst_h, dst_w), continuous column coords
    src_row_frac: np.ndarray   # (dst_h, dst_w), continuous row coords
    valid: np.ndarray          # (dst_h, dst_w), bool
```

- `src_col_frac` / `src_row_frac` are in **source pixel coordinate space** (0,0 = top-left corner of pixel). For `interpn`, grid points are at integer indices, so coordinates relative to the enlarged window origin map directly to the interpolator without additional offset adjustment.
- `valid` marks destination pixels whose kernel support window falls entirely inside the source array. Pixels on the border of the source image, even with halo expansion, may still be invalid if the kernel radius overruns the edge; these are filled with `nodata`.

### Halo sizing table

| Method    | Radius | `interpn` class                |
|-----------|--------|--------------------------------|
| nearest   | 0      | `None` (indexing fast path)    |
| bilinear  | 1      | `interpn.MultilinearRegular`   |
| cubic     | 2      | `interpn.MulticubicRegular`    |

> Note: `interpn` does not currently implement Lanczos or other sinc-based kernels. If those are needed in the future, they can be added behind the same kernel abstraction using either a different backend or a manual implementation.

## Integration Points

### `_core.py` / `open()`

- Accept `resampling: str = "nearest"` and `overview_level: int | None = None`.
- Validate `resampling` against `_KERNELS` early; fail fast.
- Pass both through to `_build_dataarray` and `MultiBandStacBackendArray`.

### `_backend.py` / `MultiBandStacBackendArray`

- Store `resampling_kernel: ResamplingKernel` and `overview_level: int | None` as fields.
- Forward them in `_ChunkReadPlan`.
- In `_resolve_spatial_window`, expand the chunk's destination-CRS bbox by `kernel.radius * abs(chunk_affine.a)` before projecting to EPSG:4326. This expanded bbox becomes `chunk_bbox_4326` in the read plan, ensuring the DuckDB search returns items that are adjacent but just outside the strict geometric boundary.

### `_chunk_reader.py`

- `_select_overview` gains an `overview_level: int | None` parameter:
  - If `None`, keep existing logic.
  - If an integer, select that exact overview (or raise if out of range).
- `_native_window` gains `halo: int` and expands the window accordingly.
- `_read_item_band` passes `kernel.radius` to `_native_window`.
- In `_apply_bands_with_warp_cache`, branch on `kernel.name == "nearest"`:
  - nearest: use existing `compute_warp_map` + `apply_warp_map` (fast integer path).
  - other: use `compute_interp_map` + `apply_interp_map`, which constructs the `interpn` interpolator and evaluates at the shifted fractional coordinates.

### `_reproject.py`

- Keep `compute_warp_map` / `apply_warp_map` untouched for the nearest-neighbor fast path.
- Add `compute_interp_map` and `apply_interp_map`.
- `apply_interp_map` constructs the appropriate `interpn` interpolator from the enlarged source array and evaluates it at the valid fractional coordinates (offset by the halo window origin). Invalid pixels are filled with `nodata`.

## Migration Path

1. Phase 1 (this spec): add `resampling="nearest"` (default) and `overview_level` to `open()`. Implement bilinear and cubic via `interpn`. No breaking changes.
2. Phase 2: any additional kernels (e.g. Lanczos) can be added later when a suitable backend is identified.

## Testing Strategy

- **Unit tests** in `test_reproject.py`:
  - Round-trip test: create a synthetic affine + CRS, project a known checkerboard array, assert interpolated values match expected sums/weights.
  - Boundary test: verify that destination pixels whose kernel extends past the source edge receive `nodata`.
  - `interpn` smoke test: verify that `apply_interp_map` produces the same result as `interpn` evaluated directly on a known grid.
- **Unit tests** in `test_chunk_reader.py`:
  - Mock `GeoTIFF` / `Overview` with known dimensions; assert `_native_window` with `halo=1` returns a window expanded by 1 on each side.
- **Integration tests**:
  - Open a local COG at various `resampling` settings and compare mean absolute difference against `rasterio.warp.reproject` with the same method on a small region (allowing for minor differences between `interpn` cubic and scipy spline cubic).
- **Regression**:
  - Ensure `resampling="nearest"` produces byte-identical output to the pre-spec behavior.

## Decision Log

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| Keep nearest-neighbor as default | Switch default to bilinear | Backward compatibility; nearest is fastest and adequate for categorical data. |
| Separate `overview_level` parameter instead of overloading `resampling` | Encode overview policy inside the resampling string | Clean separation of concerns; one parameter controls interpolation quality, the other controls I/O level. |
| Use `interpn` for non-nearest resampling | scipy `map_coordinates`, or pure-numpy fallback | `interpn` is faster, allocation-free, and has better floating-point accuracy than scipy. It is also lighter to depend on (only numpy). |
| `interpn` as required rather than optional dep | Make `interpn` optional; fall back to pure-numpy or nearest | The package is small (Rust extension, no transitive deps), provides the exact functionality we need, and avoids maintaining multiple interpolation backends. |
| Halo size derived from kernel table | Single fixed halo for all non-NN methods | Cubic needs a larger halo than bilinear; a fixed halo would over-fetch for bilinear and under-fetch for cubic. |
| Do not validate overview resampling method | Reject overview usage when it conflicts with user resampling | The provider controls overview generation; lazycogs cannot know the method. Documenting the interaction is more honest than guessing. |
| Expand WGS84 search bbox by halo distance | Expand only source read window, keep search bbox exact | Without expanding the search bbox, adjacent items that contribute halo pixels are never discovered, leading to edge artifacts at tile boundaries. |
| Derive halo distance from `radius * chunk_resolution` | Require per-item halo distance computation at search time | Per-item computation is impossible before the search. Using chunk resolution is conservative for fine sources and adequate for typical use. |
| InterpMap as separate dataclass | Reuse WarpMap with float dtypes | WarpMap semantics are integer indices + direct indexing; interpolation needs fractional coords and a different apply function. Keeping them separate avoids subtle bugs. |

## Open Questions

1. **Overview/resampling conflict:** when a user requests `resampling="cubic"` at 120 m from 30 m native data, and the COG contains a nearest-neighbor 120 m overview, should lazycogs warn the user that the overview method conflicts with their requested resampling? Or is the `overview_level` escape hatch sufficient?
2. **Lanczos implementation:** `interpn` does not currently support Lanczos or other sinc-based kernels. Should we defer Lanczos entirely, or add a secondary backend (e.g. `scipy.ndimage.map_coordinates` with a custom kernel) later?
3. **Performance of pure-numpy nearest vs. interpn bilinear:** should we benchmark to see if the fast path difference justifies keeping nearest as the unconditional default, or if bilinear is fast enough to be the default for continuous data?
4. **Upsampling halo distance:** when the output resolution is finer than the source (e.g. 10 m output from 30 m source), `radius * chunk_resolution` may under-estimate the ground distance of the source halo. Should we add a `halo_pixels` override so users can enforce a larger search radius in upsampling workflows?
5. **Internal nodata handling with interpolation:** `interpn` interpolates all source values, including internal nodata pixels. Should lazycogs detect nodata regions within the source window and mask the interpolated result, or document this as a known limitation?
6. **Internal tile boundaries within a single COG:** if a COG is tiled and we expand the read window by a halo, `async_geotiff.read(window=...)` fetches all tiles intersecting the window. Are there unexpected performance cliffs when the halo pushes us into adjacent internal tiles?

## Status

- [x] Designing
- [ ] Approved — ready to plan
- [ ] Implementing
- [ ] Implemented

## References

- `_reproject.py` — current nearest-neighbor warp map implementation
- `_chunk_reader.py` — overview selection (`_select_overview`) and window computation (`_native_window`)
- `_backend.py` — `MultiBandStacBackendArray` and `_ChunkReadPlan`
- `_core.py` — `open()` public API
- [interpn](https://github.com/jlogan03/interpn) — Rust/Python interpolation library used for bilinear and cubic
- COG specification — overview generation is producer-controlled
