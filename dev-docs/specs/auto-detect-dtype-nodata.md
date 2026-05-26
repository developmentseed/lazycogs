# Spec: Auto-detect dtype and define nodata semantics in `lazycogs.open()`

## Context

The concurrency refactor described in `TECH_DEBT.md` has landed. `open()` no longer needs to invent a short-lived event loop for startup inspection work; sync callers can reuse the shared lazycogs loop via `run_on_loop(...)`.

That makes it a good time to revisit `dtype` and `nodata` defaults together.

Today `lazycogs.open()` still defaults the output array dtype to `float32` and the output `nodata` to `None` unless the caller overrides them. At chunk-read time, lazycogs already consults the COG's internal `geotiff.nodata` when `ctx.nodata` is `None`, so internal masking is often more correct than the output array contract. The gap is at the xarray boundary: callers cannot reliably inspect the returned `DataArray` and know what dtype or nodata semantics it represents.

This spec does two things:

1. Auto-detect the default output dtype from sample COGs.
2. Make the output nodata contract explicit and testable.

## Goals

- Infer the output `dtype` from sample COGs when the caller does not pass `dtype=`.
- Infer a single output `nodata` value when the caller does not pass `nodata=` and the sampled bands support a coherent scalar sentinel.
- Keep `open()` latency acceptable by sampling one matching item and opening one COG per requested band concurrently.
- Reject incompatible configurations such as float-required mosaic methods with an inferred integer dtype.
- Document exactly what the returned `DataArray` means when nodata is known, conflicting, or unknown.

## Non-goals

- Per-band dtype in the returned `DataArray`.
- Per-band nodata metadata in the returned `DataArray`.
- Returning masked xarray arrays instead of plain numeric arrays.
- Changing explicit `dtype=` or `nodata=` overrides.
- Solving semantic cloud masking or QA masking.

## Current behavior

The current implementation has three different nodata layers:

1. `open()` stores one scalar `nodata` value on the backend plan, defaulting to `None`.
2. `_chunk_reader.py` computes `effective_nodata = ctx.nodata if ctx.nodata is not None else geotiff.nodata` per asset read.
3. Internal mosaic methods operate on `MaskedArray`s built from `effective_nodata`.

A few consequences matter here:

- Source COG nodata is already respected for internal masking when the caller does not override it.
- Empty chunks and out-of-bounds reprojection pixels are filled with `ctx.nodata` when known, otherwise `0`.
- The returned `DataArray` does not currently expose a clear nodata contract.
- `float32` remains the output dtype default even for obviously integer collections.

## Proposed output contract

### 1. Returned arrays stay plain numeric arrays

`lazycogs.open()` should continue to return a normal numeric `xr.DataArray`, not a masked array. Internal masking remains an implementation detail used while mosaicking overlapping inputs.

### 2. Nodata is a single scalar output sentinel when it is knowable

When lazycogs can determine one scalar output nodata value, that value means:

- uncovered pixels in empty chunks
- out-of-bounds pixels introduced by reprojection
- pixels that remain invalid after mosaicking because every contributing input pixel was nodata

When nodata is known, the returned `DataArray` should advertise it via:

- `attrs["_FillValue"]`
- `encoding["_FillValue"]`

This keeps the contract lean and aligns with downstream serialization tools such as rioxarray.

### 3. Auto-detected nodata must be coherent, not guessed

If the caller omits `nodata=`:

- If all sampled non-`None` nodata values are identical, use that value.
- If all sampled nodata values are `None`, keep output nodata as `None`.
- If sampled bands disagree on non-`None` nodata values, raise `ValueError` and require the caller to pass `nodata=` explicitly.

Do not silently pick the first nodata value. The output array only has room for one scalar sentinel, so conflicting sampled values are a real contract problem, not a warning-level detail.

### 4. `nodata=None` remains allowed, but its meaning is narrow

If output nodata is `None`, lazycogs is saying:

- it does not know a scalar nodata sentinel for the returned array
- no nodata metadata is attached to the `DataArray`
- `0` may still appear in uncovered regions as an implementation fill value

That last bullet is ugly but honest. In this mode, `0` is **not** declared as semantic nodata. If callers need a stable sentinel for downstream analysis, they must pass `nodata=` explicitly.

This keeps the current runtime behavior while finally documenting it clearly.

## Proposed dtype contract

### Explicit override wins

If the caller passes `dtype=`, lazycogs uses it exactly as today.

### Auto-detect when omitted

If `dtype` is omitted, `open()` samples one COG per requested band from the first matching item and promotes the sampled dtypes to one safe output dtype.

### Promotion rules

`_promote_dtypes()` should obey these rules in order:

1. All identical -> return that dtype.
2. Any float present -> return `float32`, or `float64` if any sampled dtype is `float64`.
3. Mixed integer widths -> use the wider width when one dtype can safely contain the other.
4. Mixed unsigned/signed integers -> promote to the smallest signed dtype that can represent both ranges.
5. `uint64 + int64` -> raise `ValueError` because there is no wider integer type.

Selected examples:

| Unsigned | Signed | Promoted |
|---|---|---|
| `uint8` | `int8` | `int16` |
| `uint8` | `int16` | `int16` |
| `uint16` | `int16` | `int32` |
| `uint32` | `int32` | `int64` |
| `uint64` | `int64` | `ValueError` |

## Startup inspection design

The existing `_discover_bands` and `_smoketest_store` split should be replaced by one coherent inspection helper.

### Internal model

```python
@dataclass(frozen=True)
class _ItemInspection:
    bands: list[str]
    dtypes: dict[str, np.dtype]
    nodata_values: dict[str, float | None]
    item_found: bool
```

### Async helper

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
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> _ItemInspection:
    ...
```

### Sync wrapper

`open()` remains sync, so it should call a thin sync wrapper that uses `run_on_loop(...)` to execute `_inspect_first_item_async(...)` on the shared background loop.

Do not use `asyncio.run` here. That was reasonable before the concurrency cleanup, but it is the wrong fit for the current architecture.

### Inspection steps

1. Query DuckDB for the first matching item.
2. If no item matches, return `item_found=False`.
3. Determine band order using the same current preference for `roles=["data"]` or TIFF-like assets.
4. Restrict to caller-provided `bands=` when present.
5. Resolve one asset HREF per requested band.
6. Resolve the store/path for each HREF.
7. Concurrently `await GeoTIFF.open(path, store=resolved_store)` for all requested bands.
8. Extract `geotiff.dtype` and `geotiff.nodata`.
9. Return `_ItemInspection`.

This preserves the current fail-early store validation while eliminating redundant startup queries.

## `open()` flow after the change

```text
open()
  └── _inspect_first_item()          # replaces _discover_bands + _smoketest_store
        ├── one DuckDB first-item query
        └── concurrent GeoTIFF metadata opens on the shared loop
  ├── resolve output bands
  ├── resolve output dtype
  ├── resolve output nodata
  ├── validate mosaic method vs dtype
  ├── _build_time_steps()
  └── _build_dataarray()
```

Resolution rules inside `open()`:

- `bands`
  - explicit `bands=` wins
  - otherwise use `inspection.bands`
- `dtype`
  - explicit `dtype=` wins
  - otherwise `out_dtype = _promote_dtypes(...)`
- `nodata`
  - explicit `nodata=` wins
  - otherwise `out_nodata = _resolve_output_nodata(inspection.nodata_values)`

## Mosaic-method validation

Some mosaic methods only make sense with floating-point output.

Add a class attribute:

```python
class MosaicMethodBase(ABC):
    requires_float: bool = False
```

Set `requires_float = True` on:

- `MeanMethod`
- `MedianMethod`
- `StdevMethod`

Validation in `open()`:

```python
if method_cls.requires_float and not np.issubdtype(out_dtype, np.floating):
    raise ValueError(
        f"{method_cls.__name__} requires a floating-point dtype, "
        f"but got {out_dtype}. Pass dtype='float32' or dtype='float64'."
    )
```

## Integration with current chunk-read behavior

The chunk-read path does not need a semantic rewrite.

This line stays the source of truth for per-asset masking:

```python
effective_nodata = ctx.nodata if ctx.nodata is not None else geotiff.nodata
```

That means:

- explicit user `nodata=` remains authoritative
- otherwise per-COG nodata still drives internal masking
- the new output-level nodata contract simply makes the xarray boundary honest

What changes is the startup contract and metadata, not the basic masking algorithm.

## DataArray attrs

When output nodata is known, `_build_dataarray()` should attach:

```python
da.encoding["_FillValue"] = out_nodata
```

When output nodata is `None`, no `_FillValue` encoding should be set.

This spec does not require xarray encoding changes yet.

## Failure modes

Raise `ValueError` when:

- no matching STAC items exist
- sampled dtypes cannot be promoted safely (`uint64 + int64`)
- sampled nodata values conflict across requested bands and the caller did not pass `nodata=`
- a float-required mosaic method is paired with a non-floating output dtype

## Testing strategy

### Unit tests

Add focused unit coverage for:

- `_promote_dtypes()` across integer and float combinations
- `_resolve_output_nodata()` for:
  - all `None`
  - one repeated scalar value
  - conflicting scalar values
- `_inspect_first_item_async()` with mocked DuckDB + mocked `GeoTIFF.open`
- sync wrapper using `run_on_loop(...)`

### Integration tests

Add integration coverage for:

- integer collection -> inferred integer dtype
- inferred nodata attached to `da.encoding["_FillValue"]`
- conflicting sampled nodata -> `ValueError` unless `nodata=` is passed
- `MeanMethod` with inferred integer dtype -> `ValueError`
- explicit `dtype="float32"` still works with float-required methods

### Regression expectations

Existing behavior should remain true for:

- per-COG nodata masking during mosaicking
- empty-chunk fill behavior
- reprojection fill behavior
- explicit caller overrides

## Decision log

| Decision | Rationale |
|---|---|
| Use `run_on_loop(...)` instead of `asyncio.run(...)` | Matches the post-refactor concurrency model and avoids nested-loop nonsense. |
| Replace `_discover_bands` + `_smoketest_store` with one inspection helper | One startup query, one coherent control point. |
| Raise on conflicting sampled nodata values | One output array cannot honestly advertise multiple scalar sentinels. |
| Keep returning plain numeric arrays | Preserves the current API and avoids forcing xarray masked-array semantics into the public contract. |
| Treat `0` as implementation fill when output nodata is unknown | Matches current behavior without pretending it is semantic nodata. |

## Open questions

- Should `CountMethod` force an integer output dtype when `dtype` is omitted, or should it continue to respect the generic promotion result? For now this spec leaves `CountMethod` alone.
- Should lazycogs eventually offer an explicit nodata decoding mode for in-memory analysis, rather than only advertising `_FillValue` for downstream serialization?
- If a collection has no source nodata and the user omits `nodata=`, should lazycogs eventually promote integer outputs to float and use `NaN` for uncovered pixels? That would be a larger behavioral change and is not part of this pass.

## Recommended next implementation order

1. Add `_ItemInspection`, `_inspect_first_item_async`, and the sync wrapper.
2. Add `_promote_dtypes()` and `_resolve_output_nodata()`.
3. Wire startup inspection into `open()`.
4. Add `requires_float` validation to mosaic methods.
5. Set nodata attrs in `_build_dataarray()`.
6. Update `README.md` and `ARCHITECTURE.md` once behavior lands.

## Status

- [x] Designing
- [x] Approved -- ready to implement
- [ ] Implementing
- [ ] Implemented
