# Spec: Clean Async/Sync Layering in `lazycogs`

## Context

The current main branch has a working but inconsistent async/sync layering. The chunk-read path goes through a sync→async bridge (`_run_coroutine`), but inside the async path it calls a synchronous DuckDB query inline (`_search_items`), which blocks the event loop. The xarray backend exposes only a synchronous `__getitem__`. There is no async public API after `open_async` was removed in #37 because it was a fake-async wrapper around a sync implementation.

This spec aligns the codebase with a single layering principle: **async functions are the real implementation; sync entry points are thin adapters on top.** Sync leaves (DuckDB, pyproj/numpy reprojection) are isolated and called only via `run_in_executor`.

This unblocks two downstream goals:

1. An interactive map application that asynchronously fetches chunks from the array without crossing a blocking sync→async boundary inside the read path.
2. A lower-level public async API that callers can use without going through xarray indexing (relevant to issue #31 "high-level xarray API but another low-level API with more control").

It also resolves the honesty problem flagged in #31: declaring something `async` while blocking the loop on DuckDB.

This spec does not implement xarray's `async_getitem` (#29) directly — it lays the groundwork so that change becomes a small, mechanical follow-up.

## Goals

- Make every `async` function in the read path actually yield the loop during I/O and CPU-bound work. No sync calls inline inside coroutines.
- Establish a single source of truth per operation: one real async implementation, one thin sync adapter on top.
- Isolate sync third-party calls (DuckDB queries, reprojection) at clearly-named sync leaves that the async layer wraps in `run_in_executor`.
- Use a dedicated executor for DuckDB queries, separate from the reprojection executor, so a long reprojection cannot starve a queued query.
- Update `ARCHITECTURE.md` and `_executor.py` docstrings to match current behaviour (both have minor drift from recent changes).

## Non-Goals

- Implementing `async_getitem` on `MultiBandStacBackendArray` (issue #29). This spec makes that change trivial but does not include it.
- Restoring or reintroducing `open_async`. The async open path is out of scope; `open()` remains synchronous because DuckDB at open time runs a small, fixed number of queries that don't materially benefit from async. It can be revisited later if/when DuckDB connection pooling lands.
- Real DuckDB parallelism via per-thread `DuckdbClient` pools. That depends on an open question about `rustac.DuckdbClient` internals (see #31). This spec makes the layering ready for that change but does not perform it.
- Changing the persistent per-thread background loop strategy. The loop persists across calls because async-geotiff/obstore can fire callbacks after the awaited coroutine returns. Replacing this with `asyncio.run` per call would require verifying that constraint is no longer true.
- Any change to `_chunk_reader.py`'s internal logic (mosaic method dispatch, `_drain_in_order`, semaphore-bounded fan-out). That layer is already correctly shaped.

## Constraints & Assumptions

- DuckDB's Python client serialises access on a single connection internally. Multiple coroutines calling `client.search()` concurrently on the same `DuckdbClient` are safe but not parallel. The threading.Lock was removed in #39; this spec does not reintroduce any Python-level synchronisation.
- `async-geotiff` and `obstore` may fire callbacks on background threads after a coroutine completes. The persistent per-thread background event loop in `_get_or_create_background_loop` exists specifically to handle this — the loop must outlive any single call.
- `_apply_bands_with_warp_cache` is deterministic with respect to the warp cache, so concurrent writes from different coroutines are safe (last-write-wins on identical values).
- The xarray BackendArray protocol requires a synchronous `__getitem__`. The async equivalent (`async_getitem`) is optional and will be added in a follow-up.
- Tests currently use `asyncio.run(run())` from a top-level entry point and call sync `lazycogs.open(...)` inside. The async versions of internals must be exercised through new tests.

## Architecture Overview

Target call graph (all `_search_items*` and reprojection arrows now go through `run_in_executor`):

```
PUBLIC API
==========
open(...)                          SYNC   (unchanged from today)
read_chunk(...)                    SYNC   (new — thin adapter)
    └── asyncio.run(read_chunk_async(...))
read_chunk_async(...)              ASYNC  (new — promotes async_mosaic_chunk)

XARRAY BACKEND
==============
MultiBandStacBackendArray.__getitem__       SYNC   (unchanged signature)
    └── _run_coroutine(self._async_getitem(key))
MultiBandStacBackendArray._async_getitem    ASYNC  (new — single source of truth)
    ├── resolve indexers, build _ChunkReadPlan         sync, pure CPU, inline
    └── await _read_chunk_all_dates(plan)              (renamed from _run_mosaic_all_dates)

ASYNC ORCHESTRATION
===================
_read_chunk_all_dates(time_indices, plan)   ASYNC
    └── asyncio.gather(_read_one_date(t, plan) for t in time_indices)
_read_one_date(t_idx, plan)                 ASYNC
    ├── items = await _search_items_async(plan, date)
    │       └── loop.run_in_executor(_DUCKDB_EXECUTOR, _search_items_sync, plan, date)
    └── await async_mosaic_chunk(items, ...)           (existing, unchanged)

SYNC LEAVES
===========
_search_items_sync(plan, date)              SYNC   (renamed from _search_items)
_apply_bands_with_warp_cache(...)           SYNC   (existing, unchanged)
```

Key shape changes from today:

- `_search_items` → `_search_items_sync` (renamed for clarity), wrapped by new `_search_items_async`.
- `_run_one_date` calls `_search_items_async` instead of `_search_items` directly.
- New module-level `_DUCKDB_EXECUTOR` separate from the loop's default reprojection executor.
- New `_async_getitem` method on `MultiBandStacBackendArray`. `_raw_getitem` is removed; its index-resolution logic moves into `_async_getitem` (still inline, sync, pure CPU). The xarray `explicit_indexing_adapter` call in `__getitem__` now points to a new sync shim that bridges to `_async_getitem` via `_run_coroutine`.
- New public `read_chunk_async` and `read_chunk` (thin promotions of `async_mosaic_chunk`, exported from `__init__.py`).

The same shape changes apply in `_explain.py`: the inline `backend.duckdb_client.search(...)` call becomes `await loop.run_in_executor(_DUCKDB_EXECUTOR, ...)`.

## Detailed Design

### 1. Split `_search_items` into sync leaf + async wrapper

In `src/lazycogs/_backend.py`:

- Rename existing `_search_items` to `_search_items_sync`. No body changes; this is a pure rename so the sync nature is visible at the call site.
- Add `_search_items_async`:

```python
async def _search_items_async(plan: _ChunkReadPlan, date: str) -> list[Any]:
    """Run the DuckDB search on the dedicated DuckDB executor.

    DuckDB queries serialise on a single connection internally, so this
    yields the event loop during the query but does not produce parallel
    queries against the same DuckdbClient.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_DUCKDB_EXECUTOR, _search_items_sync, plan, date)
```

- Update `_run_one_date` to call `await _search_items_async(plan, date)` instead of `_search_items(plan, date)`.

### 2. Add a dedicated DuckDB executor

In `src/lazycogs/_backend.py`, add a module-level executor:

```python
# DuckDB queries serialise on a single connection, so a small pool is enough.
# Kept separate from the reprojection executor so a long reprojection cannot
# starve a queued query within the same chunk read.
_DUCKDB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="lazycogs-duckdb",
)
```

Rationale for `max_workers=2`: one thread is sufficient for a single `DuckdbClient` (queries serialise), but a second thread provides headroom if a future change introduces a second client (e.g., for `_explain.py` running alongside a chunk read). This is intentionally small — tuning waits for the per-thread `DuckdbClient` pool work.

The executor is created at module import. It is a daemon-style pool (default for `ThreadPoolExecutor`); no explicit shutdown is wired in. If a shutdown hook becomes necessary (e.g., for clean test teardown), add it via `atexit.register`.

### 3. Rename `_run_mosaic_all_dates` → `_read_chunk_all_dates`

Pure rename for consistency with the new vocabulary (`read_chunk_async`, `_async_getitem`). Update all call sites and the docstring. No behaviour change.

### 4. Introduce `_async_getitem` on `MultiBandStacBackendArray`

This is the single source of truth for chunk reads. The synchronous `__getitem__` becomes a thin adapter.

- Rename `_raw_getitem` → `_async_getitem` and convert it to `async def`.
- The body is essentially unchanged except that `_run_coroutine(_run_mosaic_all_dates(...))` becomes `await _read_chunk_all_dates(...)`.
- Add a new private sync shim that `explicit_indexing_adapter` can call:

```python
def _sync_getitem(self, key: tuple[Any, ...]) -> np.ndarray:
    """Sync adapter that runs _async_getitem on the background loop."""
    return _run_coroutine(self._async_getitem(key))
```

- Update `__getitem__` to point `explicit_indexing_adapter` at `self._sync_getitem`:

```python
def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
    return indexing.explicit_indexing_adapter(
        key,
        self.shape,
        indexing.IndexingSupport.BASIC,
        self._sync_getitem,
    )
```

This layout means a future `async_getitem` (issue #29) is a one-liner: `return await self._async_getitem(self._adapt_key(key))`, where `_adapt_key` is whatever async-equivalent of `explicit_indexing_adapter` xarray provides. The real work is already in `_async_getitem`.

### 5. Apply the same pattern in `_explain.py`

In `_explain_async`, the inline `backend.duckdb_client.search(...)` call becomes:

```python
loop = asyncio.get_running_loop()
items = await loop.run_in_executor(
    _DUCKDB_EXECUTOR,
    lambda: backend.duckdb_client.search(
        backend.parquet_path,
        bbox=chunk_bbox_4326,
        datetime=date_filter,
        sortby=backend.sortby,
        filter=backend.filter,
        ids=backend.ids,
    ),
)
```

Import `_DUCKDB_EXECUTOR` from `_backend` (or move it to a shared location like `_executor.py` if cyclic imports become a concern — `_executor.py` is the natural home).

Recommendation: move `_DUCKDB_EXECUTOR` to `_executor.py` alongside the existing reprojection executor configuration. Rename the file's docstring from "Configure the per-event-loop thread pool size for CPU-bound reprojection work" to something like "Thread pool configuration for CPU-bound and DuckDB work" to reflect the broader scope.

### 6. Promote `async_mosaic_chunk` to public `read_chunk_async`

In `src/lazycogs/_chunk_reader.py`, the existing `async_mosaic_chunk` already has the right shape for a public low-level API. Two options:

**Option A (preferred):** Rename `async_mosaic_chunk` → `read_chunk_async`. Update internal call sites (`_run_one_date` in `_backend.py`). Add a thin sync wrapper `read_chunk` in the same file that calls `_run_coroutine(read_chunk_async(...))`. Export both from `lazycogs/__init__.py`.

**Option B:** Keep `async_mosaic_chunk` as the internal name and add a `read_chunk_async` alias that's just `read_chunk_async = async_mosaic_chunk`. Less clean but avoids touching call sites.

Pick A. It's mechanical and the new name is more accurate (the function does more than mosaicking — it opens, reads, reprojects, and mosaics).

`read_chunk_async`'s signature and behaviour are unchanged from `async_mosaic_chunk`. The sync `read_chunk` wrapper has the same signature minus the awaitable return type.

### 7. Documentation updates

Three places have drift from current code:

**`ARCHITECTURE.md` line 154** ("Jupyter fallback"): the description says `_run_coroutine()` "detects this with `asyncio.get_running_loop()` and submits the coroutine to a persistent per-thread background loop". The detection branch is gone — `_run_coroutine` always submits to the background loop. Rewrite this paragraph to describe a single path: "`_run_coroutine` submits the coroutine to a persistent per-thread background loop via `run_coroutine_threadsafe`. This is the same path whether the caller is a regular Python script, a Jupyter kernel, or a dask worker. The persistent loop is required because async-geotiff and obstore may fire callbacks after the awaited coroutine returns; a fresh per-call loop would tear down before those callbacks land."

**`ARCHITECTURE.md` Phase 1 step 8**: update to reflect the renamed `_read_chunk_all_dates` and `_async_getitem`. Note that `_search_items_async` runs on the dedicated DuckDB executor.

**`src/lazycogs/_executor.py:35-39`** (docstring of `set_reproject_workers`): says "Each chunk read creates a fresh asyncio event loop with its own dedicated `ThreadPoolExecutor` bounded to `n` workers." This is no longer true — the loop is persistent per-thread, and the executor is created once per loop, not per chunk. Rewrite to describe the persistent-loop model: "Each thread (dask worker, Jupyter kernel callback thread, etc.) gets one persistent background event loop with one bounded reprojection `ThreadPoolExecutor`. All chunk reads on that thread share the same loop and executor."

### 8. Public API exports

Update `src/lazycogs/__init__.py`:

```python
from lazycogs._chunk_reader import read_chunk, read_chunk_async
```

Add both to `__all__`.

## Testing Strategy

The async layering changes are mostly mechanical, but a few specific things need direct test coverage:

- **`_search_items_async` yields the loop.** Add a test that runs two `_search_items_async` calls concurrently with `asyncio.gather` and asserts that the wall-clock time is less than the sum of individual call times by a meaningful margin (allowing for DuckDB's internal serialisation, the gain comes from the loop being free for other work). A weaker but easier check: assert the function is a coroutine and that it returns the same result as `_search_items_sync` for equivalent inputs.
- **`read_chunk_async` round-trip.** Add a test that calls `read_chunk_async(...)` directly with a small list of items (use the existing test fixtures) and verifies it returns the same dict-of-arrays as `read_chunk(...)` (the sync wrapper). Both should match what comes out of `da.isel(...).load()` on the equivalent xarray slice.
- **`_async_getitem` is awaitable from an external loop.** Add a test that constructs a `MultiBandStacBackendArray`, builds a basic key, and `await`s `backend._async_getitem(key)` from inside `asyncio.run(...)`. This is the precondition for `async_getitem` (#29) — if this works, the xarray protocol method is trivial. Compare the result to `backend[key]` for equality.
- **Existing tests still pass.** The sync `__getitem__` path goes through `_sync_getitem` → `_run_coroutine` → `_async_getitem`. All existing chunk-read behaviour (FirstMethod early exit, mosaic ordering, nodata fill, dimension squeezing) must continue to work without modification.
- **Explain still works.** Run the existing explain test suite. The `run_in_executor` change in `_explain.py` should be transparent.

No new integration test is required for the executor split — the existing integration test exercises the read path end-to-end and will fail visibly if the executor wiring breaks.

## Migration & Rollout

This is an internal refactor with one new public surface (`read_chunk` / `read_chunk_async`). All existing public functions (`open`, `MultiBandStacBackendArray.__getitem__`, `da.lazycogs.explain()`) keep their signatures and behaviour.

No deprecation warnings or staged rollout needed. Ship as a single PR.

A post-release benchmark would be useful: measure chunk wall-time on a representative workload before and after, to confirm the `run_in_executor` wrap doesn't introduce measurable overhead. The existing `tests/integration_test.py` `measure(...)` blocks are sufficient for this.

## Risks

- **`run_in_executor` overhead per query.** Adding a thread hop to every DuckDB query introduces a small latency cost (typically tens of microseconds). For chunks where DuckDB dominates wall-time this is irrelevant; for chunks where the search is sub-millisecond, the overhead could be measurable. Mitigation: keep the dedicated executor pre-warmed (default `ThreadPoolExecutor` behaviour does this after first use). If this becomes a real problem, the fallback is to wrap only when `len(time_indices) > 1` — but only do this if benchmarks show it matters.
- **Executor lifetime.** A module-level `ThreadPoolExecutor` keeps daemon threads alive for the process lifetime. This is fine for normal use but can show up in test teardown as "thread leaked" warnings depending on the test runner. Mitigation: add an `atexit.register(_DUCKDB_EXECUTOR.shutdown)` if warnings appear.
- **Breakage in `_explain.py` from the import.** If `_DUCKDB_EXECUTOR` lives in `_backend.py` and `_explain.py` imports it, but `_backend.py` already imports from `_explain.py` (check this), a circular import lands. Mitigation: place `_DUCKDB_EXECUTOR` in `_executor.py` (as recommended in §5) where both modules can import it cleanly.
- **Test flakiness from concurrent coroutines.** The `asyncio.gather` of two `_search_items_async` calls timing test is inherently race-prone. Mitigation: use a deterministic check (it returns a coroutine) and a soft timing check (`assert wall_time < 1.5 * single_call_time`) rather than asserting strict speedup.

## Open Questions

- Should `_DUCKDB_EXECUTOR` be sized at `max_workers=2` (current proposal) or just `max_workers=1`? With one client, parallelism is zero either way; the second slot only helps if a future change adds a second client. Picking 2 is harmless and forward-compatible. Confirm with the reviewer before locking it in.
- Should the public `read_chunk` / `read_chunk_async` accept a `_ChunkReadPlan`-shaped dict instead of the long parameter list? The current `async_mosaic_chunk` takes 12 parameters which is a lot for a public API. Possible follow-up: introduce a `ChunkReadParams` dataclass at the public boundary. Out of scope for this spec.
- Once this lands, the follow-up for `async_getitem` (#29) is small enough to fit in the same milestone. Should it be folded into this PR or kept separate? Recommendation: keep separate — this PR is already touching enough surface area, and the xarray async protocol details deserve their own focused review.
