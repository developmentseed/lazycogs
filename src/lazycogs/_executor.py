"""Event loop and executor ownership for lazycogs background work."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine

_REPROJECT_WORKERS_ENV = "LAZYCOGS_REPROJECT_WORKERS"
_DUCKDB_MAX_WORKERS = 1


@dataclass
class _ExecutorState:
    """Mutable singleton state for the lazycogs runtime."""

    loop: asyncio.AbstractEventLoop | None = None
    loop_thread: threading.Thread | None = None
    reproject_pool: concurrent.futures.ThreadPoolExecutor | None = None
    duckdb_pool: concurrent.futures.ThreadPoolExecutor | None = None


_STATE = _ExecutorState()
_LOCK = threading.Lock()


def _default_workers() -> int:
    """Return the default worker count: CPUs up to a cap of 4.

    Reprojection (pyproj + numpy) is memory-bandwidth-bound, not compute-bound.
    Benchmarks show diminishing returns beyond 4 concurrent threads because they
    saturate the memory bus rather than adding CPU throughput. Keep the default
    conservative.
    """
    return min(os.cpu_count() or 4, 4)


def _validate_worker_count(n: int) -> int:
    """Validate a configured worker count."""
    if n < 1:
        raise ValueError(f"worker count must be >= 1, got {n!r}")
    return n


def _reproject_worker_count() -> int:
    """Return the configured reprojection worker count."""
    value = os.getenv(_REPROJECT_WORKERS_ENV)
    if value is None:
        return _default_workers()

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{_REPROJECT_WORKERS_ENV} must be an integer, got {value!r}",
        ) from exc

    return _validate_worker_count(parsed)


def _start_background_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Create and start the shared background event loop."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True, name="lazycogs-loop")
    thread.start()
    ready.wait()
    return loop, thread


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it lazily."""
    with _LOCK:
        loop = _STATE.loop
        thread = _STATE.loop_thread
        if (
            loop is not None
            and thread is not None
            and thread.is_alive()
            and loop.is_running()
            and not loop.is_closed()
        ):
            return loop

        loop, thread = _start_background_loop()
        _STATE.loop = loop
        _STATE.loop_thread = thread
        return loop


def get_reproject_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared bounded reprojection executor."""
    with _LOCK:
        if _STATE.reproject_pool is None:
            _STATE.reproject_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=_reproject_worker_count(),
                thread_name_prefix="lazycogs-reproject",
            )
        return _STATE.reproject_pool


def get_duckdb_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared bounded DuckDB executor."""
    with _LOCK:
        if _STATE.duckdb_pool is None:
            _STATE.duckdb_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=_DUCKDB_MAX_WORKERS,
                thread_name_prefix="lazycogs-duckdb",
            )
        return _STATE.duckdb_pool


def _submit_to_loop[T](
    coro: Coroutine[object, object, T],
) -> concurrent.futures.Future[T]:
    """Submit a coroutine to the shared background loop."""
    loop = _ensure_loop()
    thread = _STATE.loop_thread
    if thread is None or not thread.is_alive() or not loop.is_running():
        coro.close()
        raise RuntimeError("lazycogs background event loop is not running")
    if thread.ident == threading.get_ident():
        coro.close()
        raise RuntimeError(
            "Cannot call sync lazycogs bridge from the lazycogs event loop thread. "
            "Await the async API directly instead.",
        )
    return asyncio.run_coroutine_threadsafe(coro, loop)


def run_on_loop[T](coro: Coroutine[object, object, T]) -> T:
    """Run ``coro`` on the shared lazycogs event loop and return its result.

    This is the supported public helper for constructing loop-bound resources
    that must live on the lazycogs background loop.
    """
    return _submit_to_loop(coro).result()


def _run_coroutine[T](coro: Coroutine[object, object, T]) -> T:
    """Run an async coroutine from sync code on the shared background loop."""
    return _submit_to_loop(coro).result()


def _reset_executor_state_for_tests() -> None:
    """Reset lazycogs executor singletons for tests."""
    with _LOCK:
        loop = _STATE.loop
        thread = _STATE.loop_thread
        reproject_pool = _STATE.reproject_pool
        duckdb_pool = _STATE.duckdb_pool

        _STATE.loop = None
        _STATE.loop_thread = None
        _STATE.reproject_pool = None
        _STATE.duckdb_pool = None

    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive():
        thread.join(timeout=1)
    if loop is not None and not loop.is_closed():
        loop.close()
    if reproject_pool is not None:
        reproject_pool.shutdown(wait=True, cancel_futures=True)
    if duckdb_pool is not None:
        duckdb_pool.shutdown(wait=True, cancel_futures=True)
