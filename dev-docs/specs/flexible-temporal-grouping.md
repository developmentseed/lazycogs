# Spec: Flexible Temporal Grouping

## Context

Issue https://github.com/developmentseed/lazycogs/issues/73 asks for hourly observations to remain distinct instead of being collapsed by the default `time_period="P1D"`. The current implementation in `src/lazycogs/_temporal.py` models each supported duration as a grouper class that returns a group key, a `rustac` `datetime=` filter string, and a day-precision time coordinate. `_core.py` discovers time steps by pulling datetimes into Python, then `_backend.py` re-queries each chunk/time step with the produced `datetime=` string.

This works for day-or-coarser buckets, but it does not scale well to sub-daily or arbitrary duration behavior unless lazycogs chooses an explicit timestamp precision contract. To keep runtime queries consistent, lazycogs will continue to use rustac's `datetime=` argument for all temporal predicates and will define bucket filters as closed intervals at second precision.

## Goals

- Preserve the existing public API for duration strings such as `"P1D"`, `"P16D"`, `"P1W"`, `"P1M"`, and `"P1Y"`.
- Add `time_period=None` for one time step per unique normalized item timestamp.
- Add a path for sub-daily duration strings, starting with `"PT1H"` and likely `"PTnH"`.
- Avoid adding a new grouper class for every temporal unit.
- Avoid mixing rustac `datetime=` temporal predicates with CQL2 temporal predicates.
- Keep existing callers' behavior unchanged by default.

## Non-goals

- Do not add one time step per item identity. Multiple items with the same timestamp still mosaic into one time step.
- Do not implement every ISO 8601 duration in the first pass.
- Do not rewrite all rustac query construction by hand.
- Do not change mosaic semantics inside a time step.

## Architecture Overview

Introduce an internal time-step model that makes the xarray coordinate and rustac `datetime=` predicate explicit.

Today the backend receives `dates: list[str]`, where each string is passed as `datetime=date`. Replace that with time-step objects:

```python
@dataclass(frozen=True)
class _TimeStep:
    coord: np.datetime64
    label: str
    datetime_filter: str
```

All runtime temporal filtering uses `datetime_filter`. Bucketed sub-daily windows use closed intervals at second precision. For example, an hourly bucket starting at midnight is represented as:

```text
2025-01-01T00:00:00Z/2025-01-01T00:59:59Z
```

The next bucket starts at `2025-01-01T01:00:00Z`, so second-precision observations are assigned to only one time step.

## API or Interface Design

Public API remains centered on `open(..., time_period=...)`:

```python
def open(
    href: str,
    *,
    time_period: str | None = "P1D",
    ...,
) -> DataArray: ...
```

Supported first-pass values:

- `"P1D"`, `"PnD"`, `"P1W"`, `"P1M"`, `"P1Y"`: preserve current behavior.
- `None`: exact timestamp grouping by unique normalized timestamp, preserving the full timestamp precision present in the item datetime.
- `"PT1H"`: hourly grouping at second precision.
- `"PTnH"`: optional in the same implementation if the fixed-window semantics are straightforward.

Internal API sketch:

```python
@dataclass(frozen=True)
class _TemporalGrouping:
    period: str | None

    def time_step_for(self, datetime_value: str) -> _TimeStepKey: ...
    def build_time_step(self, key: _TimeStepKey) -> _TimeStep: ...
```

`_core._build_time_steps()` should return `list[_TimeStep]` instead of parallel `filter_strings` and `time_coords`.

`_backend._search_items_sync()` should pass the time-step filter through rustac's `datetime=` parameter:

```python
items = duckdb_client.search(
    plan.parquet_path,
    bbox=plan.chunk_bbox_4326,
    datetime=step.datetime_filter,
    filter=plan.filter_expr,
    ...,
)
```

## Data Model

### `_TimeStep`

- `label`: stable, sortable internal label.
- `coord`: xarray `time` coordinate, using day precision for old day-or-coarser modes and sub-daily precision for instant/hourly modes.
- `datetime_filter`: rustac `datetime=` value.

### Precision contract

Duration-based sub-daily grouping is second-precision. Lazycogs should document that `PTnH` windows are closed intervals whose end is one second before the next bucket start. Fractional-second item timestamps are not guaranteed to be bucketed without overlap or omission under duration-based sub-daily grouping.

`time_period=None` is different: it is exact timestamp grouping. It should preserve the full normalized timestamp, including fractional seconds when present, and use an exact rustac `datetime=` value for the runtime query.

## Integration Points

- `src/lazycogs/_temporal.py`: owns parsing, grouping keys, timestamp normalization, and creation of `_TimeStep` objects.
- `src/lazycogs/_core.py`: discovers unique time steps and passes them into `_build_dataarray()`.
- `src/lazycogs/_backend.py`: uses each `_TimeStep.datetime_filter` to query items for a chunk/time slice.
- `README.md` and `ARCHITECTURE.md`: document `None`, hourly grouping, and the new internal predicate model.

## Migration Path

1. Add `_TimeStep` while keeping current groupers emitting equivalent `datetime_filter` values.
2. Update `_core.py` and `_backend.py` to consume `_TimeStep` objects.
3. Add `time_period=None` using exact normalized timestamps that preserve full timestamp precision.
4. Add `PT1H` using second-precision closed `datetime=` ranges.
5. Optionally refactor the class-per-period grouper hierarchy into a parsed generic grouping implementation.
6. Later, optimize time-step discovery with DuckDB grouping if needed, while preserving the `datetime=` runtime predicate contract.

## Testing Strategy

- Unit-test timestamp normalization and lexicographic sort invariants.
- Unit-test `_TimeStep` generation for existing durations to prove no behavior regression.
- Unit-test `time_period=None` with multiple timestamps on the same day.
- Unit-test `time_period=None` with fractional-second timestamps to verify the full normalized timestamp is preserved.
- Unit-test `PT1H` boundary behavior so `01:00:00Z` belongs only to the next hour.
- Unit-test that hourly filters use second-precision closed ranges, e.g. `00:00:00Z/00:59:59Z`.
- Core/backend tests should verify that `datetime=step.datetime_filter` and the user-provided `filter=` are both sent unchanged to `duckdb_client.search()`.
- Documentation examples should cover hourly data and exact timestamp grouping.

## Decision Log

| Decision | Options Considered | Rationale |
| --- | --- | --- |
| Represent runtime predicates per time step | Continue passing raw strings as `dates`; introduce `_TimeStep`; add CQL2 temporal predicates | `_TimeStep` makes coordinates and `datetime=` filters explicit without changing rustac query style. |
| Use rustac `datetime=` for all temporal filters | Mix `datetime=` and CQL2; use CQL2 for everything; use `datetime=` for everything | A single predicate path is easier to reason about, preserves rustac's STAC datetime semantics, and avoids CQL2 filter composition complexity. |
| Use second precision for sub-daily duration buckets | Half-open CQL2; nanosecond/subsecond boundary math; second-precision closed intervals | Second precision is simple, matches rustac's partial-date expansion style, and avoids adjacent bucket duplication for second-precision observations. |
| Preserve duration-string API | New parameter; replace `time_period`; accept callbacks | Existing users already understand `time_period`; preserving it avoids API churn. |
| Start with `None` and hourly support | Full ISO 8601 support immediately; only `None`; only `PT1H` | This directly addresses the issue while avoiding an underspecified date-time framework. |

## Open Questions

- Should `PTnH` be included in the first implementation or should the first pass support only `PT1H`?
- What NumPy precision should instant coordinates use to preserve full timestamps reliably, likely `datetime64[ns]`?
- What NumPy precision should duration-based sub-daily coordinates use: `datetime64[s]` for the documented grouping precision, or `datetime64[ns]` for consistency with instant grouping?
- How should items with only `start_datetime` and `end_datetime` be treated for exact and hourly grouping?

## References

- GitHub issue: https://github.com/developmentseed/lazycogs/issues/73
- Existing plan: `dev-docs/plans/2026-06-23-001-feat-subdaily-and-instant-temporal-grouping-plan.md`
- Current implementation: `src/lazycogs/_temporal.py`, `src/lazycogs/_core.py`, `src/lazycogs/_backend.py`
