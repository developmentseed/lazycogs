# Spec: Auto-Detect dtype and nodata from Sample COGs in `lazycogs.open()`

## Context

Currently `lazycogs.open()` defaults to `float32` for both output `dtype` and `None` for `nodata`. The actual COG's internal nodata is used at chunk-read time if the user doesn't provide one, but the output DataArray is still `float32`. This is wasteful when all requested bands are integer types (e.g. `uint8` or `int16`). Users can override these defaults, but they must inspect the collection manually.

The library already queries the first matching item for band discovery (`_discover_bands`) and storage smoketesting (`_smoketest_store`). We can piggyback on this by opening one COG per requested band via `GeoTIFF.open` and extracting its dtype and nodata metadata with minimal overhead.

This spec replaces the internal `_discover_bands` / `_smoketest_store` split with a unified inspection helper and defines deterministic dtype promotion rules for multi-band reads.

## Goals

- Automatically infer the output `dtype` from sample COGs when the caller does not explicitly pass `dtype=`.
- Automatically infer the output `nodata` value from sample COGs when the caller does not explicitly pass `nodata=`.
- Choose a single output dtype that can represent all requested bands without overflowing or losing precision.
- Keep `open()` latency acceptable (the sample hit should be done concurrently per-band from a single item).
- Reject incompatible configurations (`MeanMethod` / `MedianMethod` / `StdevMethod` paired with an integer dtype) with a clear `ValueError`.

## Non-Goals

- Per-band dtype in the output DataArray (xarray requires a single dtype).
- Changing the existing explicit `dtype=` or `nodata=` behaviour (these remain user overrides).
- Sampling multiple items statistically (one item per band is enough for 99% of collections).

## Constraints & Assumptions

- `async_geotiff.GeoTIFF.open` is available and fast enough for metadata-only access (no pixel reads).
- The first matching STAC item is representative of the collection's data types. Mixed-dtype collections are rare enough that sampling one item is sufficient.
- The asyncio event loop isn't running inside the main thread at `open()` time, so we'll need either synchronous metadata helpers or a short-lived event loop.
- The existing `_smoketest_store` and `_discover_bands` calls already hit the first item; we should merge these into a single helper to avoid redundant queries.

## Architecture Overview

```
open()
  └── _inspect_first_item()          [replaces _discover_bands + _smoketest_store]
        ├── DuckDB: first item query
        ├── per-band: _resolve_store + GeoTIFF.open (concurrent)
        └── returns _ItemInspection
  ├── dtype inference
  │     └── _promote_dtypes()
  ├── nodata inference
  ├── mosaic method / dtype validation
  ├── _build_time_steps()
  └── return DataArray
```

A single inspection call yields:

1. Band order (data bands first, restricted by caller-provided `bands=`).
2. Sampled dtype per band.
3. Sampled nodata per band.
4. A store-connectivity flag.

This replaces two separate internal helpers (`_discover_bands` and `_smoketest_store`) with one coherent pass, eliminating a redundant DuckDB query.

## API / Interface Design

### Public API — `lazycogs.open()`

No signature changes. The `dtype` and `nodata` parameters remain optional:

```python
def open(
    parquet_path: str,
    *,
    bbox: list[float] | None = None,
    datetime: str | None = None,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
    bands: list[str] | None = None,
    sortby: str | list[str | dict[str, str]] | None = None,
    dtype: str | None = None,          # still optional; now auto-detected when None
    nodata: float | None = None,       # still optional; now auto-detected when None
    tilesize: int = 512,
    resolution: float | None = None,
    resolution_unit: str = "meters",
    store: ObjectStore | None = None,
    path_from_href: Callable[[str], str] | None = None,
    mosaic_filter: MosaicFilter | None = None,
    mosaic_method: type[MosaicMethodBase] = FirstMethod,
    overviews: list[int] | None = None,
    max_items: int = 100,
    groupby_solar_day: bool = False,
) -> xr.DataArray:
    ...
```

Behavioural contract:

- If `dtype` is passed, it is used exactly as before (user override).
- If `dtype` is omitted, `open()` samples one COG per requested band, extracts each band's native dtype, and promotes them to a single safe dtype via `_promote_dtypes()`.
- If `nodata` is passed, it is authoritative.
- If `nodata` is omitted, `open()` uses the first non-`None` nodata found among the sampled band COGs. If all sampled COGs report `None`, the DataArray's nodata stays `None`.

### Internal Data Model

```python
@dataclass(frozen=True)
class _ItemInspection:
    bands: list[str]
    dtypes: dict[str, np.dtype]          # per-band sample dtype
    nodata_values: dict[str, float | None]  # per-band sample nodata
    store_check_passed: bool
    item_found: bool
```

### Internal Helpers

```python
async def _inspect_first_item_async(
    parquet_path: str,
    *,
    duckdb_client: DuckdbClient,
    bbox: list[float] | None = None,
    datetime: str | None = None,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
    bands: list[str] | None = None,
    sortby: str | list[str | dict[str, str]] | None = None,
    store: ObjectStore | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> _ItemInspection:
    ...


def _promote_dtypes(dtypes: list[np.dtype]) -> np.dtype:
    """Return a single dtype that can safely hold all input dtypes."""
    ...
```

`_inspect_first_item_async` runs these steps:

1. Query DuckDB for the first item (same as `_discover_bands`).
2. If no items found, return an empty inspection (`item_found=False`).
3. Build band order (data bands first, fallback to all keys).
4. If `bands` is provided, restrict to the intersection.
5. For each requested band, resolve the asset `href`.
6. Call `_resolve_store(href, store, path_from_href)` for each band.
7. Concurrently `await GeoTIFF.open(path, store=resolved_store)` for each band.
8. Extract `geotiff.dtype` and `geotiff.nodata` from each result.
9. Return an `_ItemInspection` with all metadata.

### Mosaic Method Contract

```python
class MosaicMethodBase(ABC):
    requires_float: bool = False
    ...
```

`MeanMethod`, `MedianMethod`, and `StdevMethod` override `requires_float = True`.

During `open()`, after dtype is resolved:

```python
if mosaic_method_cls.requires_float and not np.issubdtype(out_dtype, np.floating):
    raise ValueError(
        f"{mosaic_method_cls.__name__} requires a floating-point dtype, "
        f"but got {out_dtype}. Pass dtype='float32' or dtype='float64' explicitly."
    )
```

### Dtype Promotion Rules

`_promote_dtypes` must obey these rules, in order:

1. **All identical** -> return that dtype (zero overhead).
2. **Any float present** -> return `float32`, or `float64` if `float64` is present.
3. **Mixed unsigned/signed integers** -> promote to signed with one extra bit when necessary.
4. **Mixed integer widths** -> use the wider width.

Promotion table (selected cases):

| Unsigned | Signed | Promoted |
|----------|--------|----------|
| uint8    | int8   | int16    |
| uint8    | int16  | int16    |
| uint16   | int16  | int32    |
| uint32   | int32  | int64    |
| uint64   | int64  | `ValueError` (overflow) |
| uint8    | int64  | int64    |
| uint16   | uint32 | uint32   |
| int8     | int16  | int16    |

Rationale for blocking `uint64 + int64`: there is no wider signed integer type. Promoting to `float64` silently would lose integer semantics. We raise `ValueError` so the caller must explicitly opt into `float64` or choose a different representation.

## Integration Points

### `_chunk_reader.py`

The existing fallback logic remains unchanged:

```python
effective_nodata = ctx.nodata if ctx.nodata is not None else geotiff.nodata
```

When the user provides `nodata`, it is authoritative at read time. When they don't, the per-COG nodata is consulted. The DataArray default nodata (now auto-detected) is used as the initial fill value in `_raw_getitem` and `async_mosaic_chunk`.

Note: `async_mosaic_chunk` fills uncovered chunks with `0` when `nodata is None`. This is still safe because those pixels represent "no items covered this chunk," not "the COG declared these as nodata."

### DuckDB Client

The inspection helper reuses the same DuckDB search pattern as `_discover_bands`. No query format changes.

### `async_geotiff`

Metadata-only reads via `GeoTIFF.open` + reading `.dtype` and `.nodata`. No pixel buffers are fetched.

## Migration Path

This is an internal refactor. `_discover_bands` and `_smoketest_store` are replaced by `_inspect_first_item` (sync wrapper) and deleted from the codebase. Callers of `open()` see no breaking change: explicit `dtype=` and `nodata=` behave exactly as before. Callers who previously relied on defaults will now see integer dtypes when the sampled COGs are integer, and more accurate nodata values.

## Testing Strategy

### Unit Tests — `test_dtype_promotion.py`

- `_promote_dtypes` for every pairwise combination of `uint8/16/32/64`, `int8/16/32/64`, `float32`, `float64`.
- Edge cases: single dtype list, empty list, overflowing `uint64 + int64`.

### Unit Tests — `_inspect_first_item`

- Mock DuckDB response + mock `GeoTIFF.open` coroutines.
- Verify concurrent opens (one per band).
- Verify `store_check_passed` flag.
- Verify empty result when no items match.

### Integration Tests

- `open(..., mosaic_method=MeanMethod)` with auto-detected `int16` raises `ValueError`.
- `open(..., mosaic_method=FirstMethod)` with auto-detected `int16` succeeds.
- Real-world parquet: open a COG collection without passing `dtype` or `nodata`; assert the resulting DataArray has the expected dtype and nodata.
- Update existing tests that assert `dtype="float32"` to accept the new auto-detected values or pass `dtype="float32"` explicitly.

### Regression Tests

- All existing tests continue to pass.

## Decision Log

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| Merge band discovery, smoketest, and metadata sampling into one helper (Option B) | Option A: extend `_smoketest_store` only | Avoids redundant DuckDB round-trips and keeps the flow coherent. `_smoketest_store` only touched one band anyway. |
| Custom integer promotion instead of `numpy.result_type` | Use `np.result_type` directly | Numpy promotes `uint16 + int16 -> float64`, which is overly conservative. A custom table preserves integer semantics in more cases. |
| Raise `ValueError` on `uint64 + int64` overflow | Silently promote to `float64` | Losing integer semantics silently is dangerous. The error forces an explicit `dtype=` override. |
| One item per band is sufficient | Statistical sampling across many items | Metadata-only COG opens are fast. Mixed-dtype collections within one band are extremely rare. One sample is the right latency/accuracy trade-off. |
| `requires_float` class attribute on `MosaicMethodBase` | Method-specific validation inside `open()` | A declarative attribute keeps `open()` logic generic and makes it easy for users to write custom mosaic methods that self-declare float requirements. |
| `asyncio.run` inside `open()` for concurrent metadata opens | Sequential sync opens | Concurrent opens cut latency linearly with band count. `open()` is typically a top-level call with no pre-existing event loop. |

## Open Questions

- **Jupyter event loop**: `asyncio.run` inside `open()` may clash with Jupyter's running event loop. If this surfaces, we may need to detect `get_running_loop()` and fall back to sequential opens or a thread-pool wrapper.
- **Per-item dtype variance**: If the first item is `uint8` but later items in the same collection are `int16`, the DataArray may overflow. This is documented as an assumed risk; users can override with `dtype=`.
- **Inspection caching**: Should repeated `open()` calls on the same parquet (different bboxes) cache the `_ItemInspection` result? Out of scope for the initial implementation, but a potential future optimization.

## Status

- [ ] Designing
- [ ] Approved -- ready to plan
- [ ] Implementing
- [ ] Implemented

## References

- Plan document: `dev-docs/plans/dtypes.md`
