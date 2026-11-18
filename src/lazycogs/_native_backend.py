"""Lazy native-grid access to already-inspected GeoTIFF assets."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from async_geotiff import Window
from xarray.backends.common import BackendArray
from xarray.core import indexing

from lazycogs._executor import run_on_loop

if TYPE_CHECKING:
    from async_geotiff import GeoTIFF


class NativeBackendArray(BackendArray):
    """Read selected bands and bounded spatial windows without reprojection.

    A single GeoTIFF represents all bands of a COG; multiple GeoTIFFs represent
    single-band item assets. Keeping the opened headers also keeps their stores
    alive for the lifetime of this array.
    """

    def __init__(self, sources: tuple[GeoTIFF, ...], *, multiband: bool) -> None:
        self.sources = sources
        self.multiband = multiband
        first = sources[0]
        self.shape = (
            first.count if multiband else len(sources),
            first.height,
            first.width,
        )
        dtypes = []
        for source in sources:
            dtype = source.dtype
            if dtype is None:
                raise ValueError("Unsupported GeoTIFF sample dtype.")
            dtypes.append(dtype)

        self.dtype = np.result_type(*dtypes)
        if np.issubdtype(self.dtype, np.floating):
            precision = np.finfo(self.dtype).nmant + 1
            # NumPy considers int64 -> float64 safe even though it can round.
            if any(
                np.issubdtype(dtype, np.integer)
                and int(np.iinfo(dtype).max).bit_length() > precision
                for dtype in dtypes
            ):
                raise ValueError(
                    f"Cannot losslessly stack asset dtypes {dtypes!r} as {self.dtype}. "
                    "Choose compatible assets with bands= or open them separately.",
                )

    def __copy__(self) -> NativeBackendArray:
        """Share the immutable backend and its stores across xarray copies."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> NativeBackendArray:
        """Avoid copying store and TIFF resources when xarray copies data."""
        memo[id(self)] = self
        return self

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        """Read a basic or xarray-adapted index synchronously."""
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._sync_getitem,
        )

    async def async_getitem(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        """Read a basic or xarray-adapted index asynchronously."""
        return await indexing.async_explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._async_getitem,
        )

    def _sync_getitem(self, key: tuple[Any, ...]) -> np.ndarray:
        return run_on_loop(self._async_getitem(key))

    async def _async_getitem(  # noqa: C901, PLR0912
        self,
        key: tuple[Any, ...],
    ) -> np.ndarray:
        selections = []
        scalar_axes = []
        for axis, (index, size) in enumerate(zip(key, self.shape, strict=True)):
            if isinstance(index, (int, np.integer)):
                pos = int(index)
                if not 0 <= pos < size:
                    raise IndexError(f"index {index} is out of bounds for axis {axis}")
                selections.append(range(pos, pos + 1))
                scalar_axes.append(axis)
            elif isinstance(index, slice):
                selections.append(range(*index.indices(size)))
            else:
                positions = np.asarray(index, dtype=np.intp)
                if np.any((positions < 0) | (positions >= size)):
                    raise IndexError(f"index is out of bounds for axis {axis}")
                selections.append(positions)

        bands, rows, cols = selections
        shape = (len(bands), len(rows), len(cols))
        if not all(shape):
            result = np.empty(shape, dtype=self.dtype)
        else:
            row_start, row_end = (
                (min(rows[0], rows[-1]), max(rows[0], rows[-1]) + 1)
                if isinstance(rows, range)
                else (int(rows.min()), int(rows.max()) + 1)
            )
            col_start, col_end = (
                (min(cols[0], cols[-1]), max(cols[0], cols[-1]) + 1)
                if isinstance(cols, range)
                else (int(cols.min()), int(cols.max()) + 1)
            )
            window = Window(
                col_start,
                row_start,
                col_end - col_start,
                row_end - row_start,
            )
            if self.multiband:
                raster = await self.sources[0].read(window=window)
                data = np.asarray(raster.data)
                if isinstance(bands, range) and bands.step == 1:
                    result = data[bands.start : bands.stop]
                else:
                    result = np.take(data, list(bands), axis=0)
            else:
                unique_bands = list(dict.fromkeys(bands))
                rasters = await asyncio.gather(
                    *(self.sources[band].read(window=window) for band in unique_bands),
                )
                data = dict(zip(unique_bands, rasters, strict=True))
                if len(bands) == 1:
                    result = np.asarray(data[bands[0]].data)
                else:
                    result = np.stack(
                        [np.asarray(data[band].data)[0] for band in bands],
                    )
            if not (isinstance(rows, range) and rows.step == 1):
                result = np.take(result, np.asarray(rows) - row_start, axis=1)
            if not (isinstance(cols, range) and cols.step == 1):
                result = np.take(result, np.asarray(cols) - col_start, axis=2)
            result = result.astype(self.dtype, copy=False)
        return np.squeeze(result, axis=tuple(scalar_axes)) if scalar_axes else result
