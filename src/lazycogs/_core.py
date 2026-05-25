"""Entry point for opening a STAC collection as a lazy xarray DataArray."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from async_geotiff import GeoTIFF
from pyproj import CRS, Transformer
from rasterix import RasterIndex
from rustac import DuckdbClient
from xarray import Coordinates, DataArray, Variable
from xarray.core import indexing

from lazycogs._backend import MultiBandStacBackendArray
from lazycogs._cql2 import _extract_filter_fields, _sortby_fields
from lazycogs._executor import run_on_loop
from lazycogs._grid import compute_output_grid
from lazycogs._mosaic_methods import FirstMethod, MosaicMethodBase
from lazycogs._store import resolve
from lazycogs._temporal import _TemporalGrouper, grouper_from_period

if TYPE_CHECKING:
    from collections.abc import Callable

    from arro3.core import Table
    from async_geotiff import Store

logger = logging.getLogger(__name__)
_INT_WIDTHS = (8, 16, 32, 64)


class _CompactDateArray(np.ndarray):
    """Numpy datetime64 array subclass with a compact display for xarray HTML repr."""

    def __new__(cls, values: np.ndarray) -> Self:
        return np.asarray(values, dtype="datetime64[D]").view(cls)

    def __str__(self) -> str:
        arr = self.view(np.ndarray)
        n = len(arr)
        if n == 1:
            return str(arr[0])
        return f"{arr[0]} \u2026 {arr[-1]} (n={n})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class _ItemInspection:
    """Sampled startup metadata from one representative STAC item."""

    bands: list[str]
    dtypes: dict[str, np.dtype]
    nodata_values: dict[str, float | int | None]
    item_found: bool


class _StoreInspectionError(RuntimeError):
    """Internal error carrying the HREF that failed startup inspection."""

    def __init__(self, href: str, original: BaseException) -> None:
        super().__init__(str(original))
        self.href = href
        self.original = original


def _ordered_bands(
    assets: dict[str, Any],
    *,
    bands: list[str] | None = None,
) -> list[str]:
    """Return asset keys in caller or preferred inspection order."""
    if bands is not None:
        missing = [band for band in bands if band not in assets]
        if missing:
            raise ValueError(
                "Requested bands "
                f"{missing!r} were not present on the first matching item.",
            )
        return bands

    data_bands: list[str] = []
    other_bands: list[str] = []
    for key, asset in assets.items():
        roles = asset.get("roles", [])
        media_type = asset.get("type", "")
        if "data" in roles or "image/tiff" in media_type:
            data_bands.append(key)
        else:
            other_bands.append(key)

    return data_bands or other_bands or list(assets)


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
    """Inspect one representative item and sample metadata for requested bands."""
    filter_fields = _extract_filter_fields(filter) if filter else set()
    items = duckdb_client.search(
        parquet_path,
        max_items=1,
        bbox=bbox,
        datetime=datetime,
        sortby=sortby,
        filter=filter,
        ids=ids,
        include=list({"assets"}.union(filter_fields).union(_sortby_fields(sortby))),
    )
    if not items:
        return _ItemInspection(
            bands=[],
            dtypes={},
            nodata_values={},
            item_found=False,
        )

    assets: dict[str, Any] = items[0].get("assets", {})
    resolved_bands = _ordered_bands(assets, bands=bands)
    if not resolved_bands:
        return _ItemInspection(
            bands=[],
            dtypes={},
            nodata_values={},
            item_found=True,
        )

    async def _inspect_band(
        band: str,
    ) -> tuple[str, np.dtype, float | int | None]:
        href = assets[band].get("href", "")
        if not href:
            raise ValueError(
                f"Asset {band!r} on the first matching item does not have an href.",
            )
        resolved_store, path = resolve(href, store=store, path_fn=path_from_href)
        try:
            geotiff = await GeoTIFF.open(path, store=resolved_store)
        except Exception as exc:
            raise _StoreInspectionError(href, exc) from exc
        return band, np.dtype(geotiff.dtype), geotiff.nodata

    results = await asyncio.gather(*[_inspect_band(band) for band in resolved_bands])
    return _ItemInspection(
        bands=resolved_bands,
        dtypes={band: dtype for band, dtype, _ in results},
        nodata_values={band: nodata for band, _, nodata in results},
        item_found=True,
    )


def _inspect_first_item(
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
    """Run representative-item inspection on the shared lazycogs event loop."""
    try:
        return run_on_loop(
            _inspect_first_item_async(
                parquet_path,
                duckdb_client=duckdb_client,
                bbox=bbox,
                datetime=datetime,
                filter=filter,
                ids=ids,
                bands=bands,
                sortby=sortby,
                store=store,
                path_from_href=path_from_href,
            ),
        )
    except _StoreInspectionError as exc:
        raise RuntimeError(
            f"Store cannot open {exc.href!r} through GeoTIFF.open: {exc.original}. "
            "Pass a configured store= argument to lazycogs.open() to authenticate. "
            "See the cloud storage guide for examples.",
        ) from exc.original


def _arrow_col(table: Table, name: str) -> list:
    """Return a column from an Arrow table as a Python list, or all-None if absent."""
    if name in table.schema.names:
        return table.column(name).to_pylist()
    return [None] * len(table)


def _promote_integer_dtypes(dtypes: list[np.dtype]) -> np.dtype:
    """Resolve sampled integer dtypes to one safe integer dtype."""
    signed = [dtype for dtype in dtypes if np.issubdtype(dtype, np.signedinteger)]
    unsigned = [dtype for dtype in dtypes if np.issubdtype(dtype, np.unsignedinteger)]

    if not signed:
        return max(unsigned, key=lambda dtype: dtype.itemsize)
    if not unsigned:
        return max(signed, key=lambda dtype: dtype.itemsize)

    max_signed_bits = max(dtype.itemsize * 8 for dtype in signed)
    max_unsigned_bits = max(dtype.itemsize * 8 for dtype in unsigned)
    required_bits = max(max_signed_bits, max_unsigned_bits + 1)
    if required_bits > max(_INT_WIDTHS):
        raise ValueError(
            "Cannot safely promote sampled dtypes containing both uint64 and int64.",
        )

    for candidate_bits in _INT_WIDTHS:
        if candidate_bits >= required_bits:
            return np.dtype(f"int{candidate_bits}")

    raise ValueError(f"Cannot safely promote sampled dtypes: {dtypes!r}")


def _promote_dtypes(dtypes: list[np.dtype]) -> np.dtype:
    """Resolve sampled dtypes to one safe output dtype."""
    normalized = [np.dtype(dtype) for dtype in dtypes]
    if not normalized:
        raise ValueError("Cannot infer dtype without at least one sampled band.")
    if len(set(normalized)) == 1:
        return normalized[0]

    if any(np.issubdtype(dtype, np.floating) for dtype in normalized):
        if any(dtype == np.dtype("float64") for dtype in normalized):
            return np.dtype("float64")
        return np.dtype("float32")

    if not all(np.issubdtype(dtype, np.integer) for dtype in normalized):
        raise ValueError(f"Unsupported sampled dtypes: {normalized!r}")

    return _promote_integer_dtypes(normalized)


def _resolve_output_dtype(
    sampled_dtypes: list[np.dtype],
    *,
    dtype: str | np.dtype | None,
    method_cls: type[MosaicMethodBase],
) -> tuple[np.dtype, bool]:
    """Resolve the output dtype, auto-promoting to float when required."""
    if dtype is not None:
        resolved = np.dtype(dtype)
        if method_cls.requires_float and not np.issubdtype(resolved, np.floating):
            raise ValueError(
                f"{method_cls.__name__} requires a floating-point dtype, "
                f"but got explicit dtype {resolved}. "
                "Pass dtype='float32' or dtype='float64', or omit dtype= to "
                "let lazycogs choose a float dtype automatically.",
            )
        return resolved, True

    resolved = _promote_dtypes(sampled_dtypes)
    if method_cls.requires_float and not np.issubdtype(resolved, np.floating):
        return np.dtype("float32"), False
    return resolved, False


def _dtype_is_compatible(source: np.dtype, resolved: np.dtype) -> bool:
    """Return True when *source* can be represented safely by *resolved*."""
    return bool(np.can_cast(source, resolved, casting="safe"))


def _resolve_output_nodata(
    nodata_values: list[float | int | None],
) -> float | int | None:
    """Resolve sampled nodata values to one scalar output sentinel."""
    saw_nan = False
    scalars: set[float | int] = set()

    for value in nodata_values:
        if value is None:
            continue
        scalar = value.item() if isinstance(value, np.generic) else value
        if isinstance(scalar, float) and np.isnan(scalar):
            saw_nan = True
            continue
        scalars.add(scalar)

    if not scalars and not saw_nan:
        return None
    if not scalars and saw_nan:
        return np.nan
    if len(scalars) == 1 and not saw_nan:
        return next(iter(scalars))

    raise ValueError(
        "Conflicting sampled nodata values; pass nodata= explicitly.",
    )


def _build_time_steps(
    parquet_path: str,
    *,
    duckdb_client: DuckdbClient,
    bbox: list[float] | None = None,
    datetime: str | None = None,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
    sortby: str | list[str | dict[str, str]] | None = None,
    temporal_grouper: _TemporalGrouper,
) -> tuple[list[str], list[np.datetime64]]:
    """Return filter strings and coordinate values for each unique time step.

    Queries *parquet_path* and buckets matching items by *temporal_grouper*.
    Only groups that have at least one matching item produce a time step, so
    the time axis never contains empty slices.

    Args:
        parquet_path: Path to a geoparquet file or hive-partitioned directory.
        duckdb_client: ``DuckdbClient`` used to query the parquet source.
        bbox: Bounding box ``[minx, miny, maxx, maxy]`` in EPSG:4326.
        datetime: RFC 3339 datetime or range to pre-filter items.
        filter: CQL2 filter expression (text string or JSON dict).
        ids: List of STAC item IDs to restrict results to.
        sortby: Sort keys forwarded to the DuckDB query.
        temporal_grouper: Grouper that maps item datetimes to group labels,
            datetime filter strings, and coordinate values.

    Returns:
        A ``(filter_strings, time_coords)`` tuple where *filter_strings* is
        the list of datetime filter strings (one per time step, sorted in
        temporal order) and *time_coords* is the corresponding list of
        ``numpy.datetime64[D]`` coordinate values.

    """
    filter_fields = _extract_filter_fields(filter) if filter else set()

    table = duckdb_client.search_to_arrow(
        parquet_path,
        bbox=bbox,
        datetime=datetime,
        sortby=sortby,
        filter=filter,
        ids=ids,
        include=list(
            {"datetime", "start_datetime"}.union(filter_fields).union(
                _sortby_fields(sortby),
            ),
        ),
    )

    keys: set[str] = set()

    if table is not None:
        logger.debug("_build_time_steps: Arrow table has %d rows", len(table))
        for dt_val, start_val in zip(
            _arrow_col(table, "datetime"),
            _arrow_col(table, "start_datetime"),
            strict=False,
        ):
            val = dt_val if dt_val is not None else start_val
            if val is None:
                continue
            iso = val if isinstance(val, str) else val.isoformat()
            keys.add(temporal_grouper.group_key(iso))

    sorted_keys = sorted(keys)
    filter_strings = [temporal_grouper.datetime_filter(k) for k in sorted_keys]
    time_coords = [temporal_grouper.to_datetime64(k) for k in sorted_keys]
    return filter_strings, time_coords


def _spatial_coords_with_eager_variables(index: RasterIndex) -> Coordinates:
    """Return RasterIndex-backed spatial coordinates with eager x/y variables.

    ``Coordinates.from_xindex(index)`` keeps the x/y coordinate variables backed
    by ``CoordinateTransformIndexingAdapter``. That works for normal access, but
    after ``DataArray.chunk(...).sel(x=..., y=..., method="nearest")`` xarray can
    end up computing scalar x/y coordinates as length-1 arrays, which then fail
    shape validation during ``compute()``.

    This helper keeps the ``RasterIndex`` itself for spatial selection semantics
    while materialising the x/y coordinate variables as plain NumPy arrays so
    scalar coordinate loads stay scalar after chunking.

    Args:
        index: Raster index describing the output grid.

    Returns:
        Coordinates containing eager x/y variables and the original RasterIndex.

    """
    index_variables = index.create_variables()
    return Coordinates(
        {
            name: (variable.dims, np.asarray(variable.data), variable.attrs)
            for name, variable in index_variables.items()
        },
        indexes=dict.fromkeys(index_variables, index),
    )


def _build_dataarray(
    *,
    parquet_path: str,
    duckdb_client: DuckdbClient,
    resolved_bands: list[str],
    filter_strings: list[str],
    time_coords: list[np.datetime64],
    bbox: tuple[float, float, float, float],
    bbox_4326: list[float],
    dst_crs: CRS,
    resolution: float,
    sortby: str | list[str | dict[str, str]] | None,
    filter: str | dict[str, Any] | None,
    ids: list[str] | None,
    nodata: float | None,
    out_dtype: np.dtype,
    dtype_was_explicit: bool,
    nodata_was_explicit: bool,
    method_cls: type[MosaicMethodBase],
    chunks: dict[str, int] | None,
    store: Store | None = None,
    max_concurrent_reads: int = 32,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Assemble the lazy DataArray from pre-computed parameters.

    This is the shared implementation used by both :func:`open` and
    the STAC search completes.

    Args:
        parquet_path: Path to a geoparquet file or hive-partitioned directory.
        duckdb_client: ``DuckdbClient`` instance passed to each
            :class:`~lazycogs._backend.MultiBandStacBackendArray` for per-chunk
        queries.
        resolved_bands: Ordered list of band/asset keys.
        filter_strings: Sorted list of ``rustac``-compatible datetime filter
            strings, one per time step.
        time_coords: ``numpy.datetime64[D]`` coordinate values corresponding
            to each entry in *filter_strings*.
        bbox: Output bounding box in ``dst_crs``.
        bbox_4326: Bounding box in EPSG:4326.
        dst_crs: Target output CRS.
        resolution: Output pixel size in ``dst_crs`` units.
        sortby: Optional rustac sort keys.
        filter: CQL2 filter expression forwarded to per-chunk DuckDB queries.
        ids: STAC item IDs forwarded to per-chunk DuckDB queries.
        nodata: No-data fill value.
        out_dtype: Output array dtype.
        dtype_was_explicit: Whether the caller passed ``dtype=`` explicitly.
        nodata_was_explicit: Whether the caller passed ``nodata=`` explicitly.
        method_cls: Mosaic method class.
        chunks: Passed to ``DataArray.chunk()`` if not ``None``.
        store: Pre-configured :class:`async_geotiff.Store` accepted by
            ``GeoTIFF.open``. When provided, it is used directly for all asset
            reads instead of resolving an obstore-backed store from each HREF.
        max_concurrent_reads: Maximum number of COG reads to run concurrently
            per chunk.
        path_from_href: Optional callable ``(href: str) -> str`` passed to
            :class:`~lazycogs._backend.MultiBandStacBackendArray`.  See
            :func:`open` for full documentation.

    Returns:
        Lazy ``xr.DataArray`` with dimensions ``(band, time, y, x)``.

    """
    dst_affine, dst_width, dst_height = compute_output_grid(
        bbox=bbox,
        resolution=resolution,
    )

    multi = MultiBandStacBackendArray(
        parquet_path=parquet_path,
        duckdb_client=duckdb_client,
        bands=resolved_bands,
        dates=filter_strings,
        dst_affine=dst_affine,
        dst_crs=dst_crs,
        bbox_4326=bbox_4326,
        sortby=sortby,
        filter=filter,
        ids=ids,
        dst_width=dst_width,
        dst_height=dst_height,
        dtype=out_dtype,
        nodata=nodata,
        dtype_was_explicit=dtype_was_explicit,
        nodata_was_explicit=nodata_was_explicit,
        mosaic_method_cls=method_cls,
        store=store,
        max_concurrent_reads=max_concurrent_reads,
        path_from_href=path_from_href,
    )
    lazy = indexing.LazilyIndexedArray(multi)
    var = Variable(("band", "time", "y", "x"), lazy)

    # Only convert to dask when the caller explicitly requests chunking.
    # Without this guard, xr.concat (used inside to_array) would eagerly load
    # LazilyIndexedArray-backed objects.  MultiBandStacBackendArray avoids that
    # concat entirely, so a narrow slice like da.isel(time=0, x=0, y=0) fetches
    # only the requested pixels.
    if chunks is not None:
        var = var.chunk(chunks)

    index = RasterIndex.from_transform(
        dst_affine,
        width=dst_width,
        height=dst_height,
        x_dim="x",
        y_dim="y",
        crs=dst_crs,
    )
    spatial_coords = _spatial_coords_with_eager_variables(index)

    time_coord = np.array(time_coords, dtype="datetime64[D]")

    gt = [
        dst_affine.a,
        dst_affine.b,
        dst_affine.c,
        dst_affine.d,
        dst_affine.e,
        dst_affine.f,
    ]

    spatial_ref = DataArray(
        np.array(0),
        attrs={
            "crs_wkt": dst_crs.to_wkt(),
            "GeoTransform": " ".join(str(v) for v in gt),
        },
    )

    attributes = {
        "grid_mapping": "spatial_ref",
        "zarr_conventions": [
            {
                "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
                "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
                "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
                "name": "proj:",
                "description": (
                    "Coordinate reference system information for geospatial data"
                ),
            },
            {
                "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
                "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
                "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
                "name": "spatial:",
                "description": "Spatial coordinate information",
            },
        ],
        "spatial:dimensions": ["y", "x"],
        "spatial:bbox": bbox,
        "spatial:transform_type": "affine",
        "spatial:transform": gt,
        "spatial:shape": [dst_height, dst_width],
        "spatial:registration": "pixel",
        "_stac_backend": multi,
        "_stac_time_coords": _CompactDateArray(time_coord),
    }

    if nodata is not None:
        attributes["nodata"] = nodata
        attributes["_FillValue"] = nodata
        attributes["missing_value"] = nodata

    # Zarr geo-proj convention
    epsg = dst_crs.to_epsg()
    if epsg is not None:
        attributes["proj:code"] = f"EPSG:{epsg}"
    else:
        attributes["proj:wkt2"] = dst_crs.to_wkt()

    return DataArray(
        var,
        coords=Coordinates(
            {"band": resolved_bands, "time": time_coord, "spatial_ref": spatial_ref},
        )
        | spatial_coords,
        attrs=attributes,
    )


def open(  # noqa: A001
    href: str,
    *,
    datetime: str | None = None,
    bbox: tuple[float, float, float, float],
    crs: str | CRS,
    resolution: float,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
    bands: list[str] | None = None,
    chunks: dict[str, int] | None = None,
    sortby: str | list[str | dict[str, str]] | None = None,
    nodata: float | None = None,
    dtype: str | np.dtype | None = None,
    mosaic_method: type[MosaicMethodBase] | None = None,
    time_period: str = "P1D",
    store: Store | None = None,
    max_concurrent_reads: int = 32,
    path_from_href: Callable[[str], str] | None = None,
    duckdb_client: DuckdbClient | None = None,
) -> DataArray:
    """Open a mosaic of STAC items as a lazy ``(band, time, y, x)`` DataArray.

    ``href`` must be a path to a geoparquet file (``.parquet`` or
    ``.geoparquet``) or, when *duckdb_client* is provided, to a
    hive-partitioned parquet directory.

    Args:
        href: Path to a geoparquet file (``.parquet`` or ``.geoparquet``)
            or a hive-partitioned parquet directory when *duckdb_client* is
            provided with ``use_hive_partitioning=True``.
        datetime: RFC 3339 datetime or range (e.g. ``"2023-01-01/2023-12-31"``)
            used to pre-filter items from the parquet.
        bbox: ``(minx, miny, maxx, maxy)`` in the target ``crs``.
        crs: Target output CRS.
        resolution: Output pixel size in ``crs`` units.
        filter: CQL2 filter expression (text string or JSON dict) forwarded
            to DuckDB queries, e.g. ``"eo:cloud_cover < 20"``.
        ids: STAC item IDs to restrict the search to.
        bands: Asset keys to include. If ``None``, inferred from the first
            matching item's preferred data assets.
        chunks: Chunk sizes passed to ``DataArray.chunk()``.  If ``None``
            (default), returns a ``LazilyIndexedArray``-backed DataArray
            where only the requested pixels are fetched on each access —
            ideal for point or small-region queries.  Pass an explicit dict
            to convert to a dask-backed array for parallel computation over
            larger regions.
        sortby: Sort keys forwarded to DuckDB queries.
        nodata: No-data fill value for output arrays. When omitted,
            lazycogs advertises a scalar nodata sentinel only when sampled
            bands agree on one.
        dtype: Output array dtype. When omitted, inferred from sampled asset
            dtypes on the first matching item. Float-only mosaic methods may
            auto-promote inferred integer outputs to ``float32``. Explicit
            integer ``dtype=`` still raises for those methods.
        mosaic_method: Mosaic method class (not instance) to use.  Defaults
            to :class:`~lazycogs._mosaic_methods.FirstMethod`.
        time_period: ISO 8601 duration string controlling how items are
            grouped into time steps.  Supported forms: ``PnD`` (days),
            ``P1W`` (ISO calendar week), ``P1M`` (calendar month), ``P1Y``
            (calendar year).  Defaults to ``"P1D"`` (one step per calendar
            day), which preserves the previous behaviour.  Multi-day windows
            such as ``"P16D"`` are aligned to an epoch of 2000-01-01.
        store: Pre-configured :class:`async_geotiff.Store` accepted by
            ``GeoTIFF.open`` to use for all asset reads. Useful when
            credentials, custom endpoints, or non-default options are needed
            without relying on automatic store resolution from each HREF. When
            ``None`` (default), each asset URL is parsed to create or reuse a
            shared cached obstore-backed store behind a small lock.
        max_concurrent_reads: Maximum number of COG reads to run concurrently
            per chunk.  Concurrency is bounded to this size with an
            ``asyncio.Semaphore``, which bounds peak in-flight memory when a
            chunk overlaps many files. Methods that support early exit (e.g. the default
            :class:`~lazycogs._mosaic_methods.FirstMethod`) will stop
            reading once every output pixel is filled, so lower values also
            reduce unnecessary I/O on dense datasets.  Defaults to 32.
        path_from_href: Optional callable ``(href: str) -> str`` that extracts
            the object path from an asset HREF.  When provided, it replaces the
            default ``urlparse``-based extraction used in
            :func:`~lazycogs._store.resolve`.  Most useful when combined with
            a custom ``store`` whose root does not align with the URL path
            structure of the asset HREFs.

            Example — NASA LPDAAC proxy https url for S3 asset::

                from obstore.store import S3Store
                from urllib.parse import urlparse

                store = S3Store(bucket="lp-prod-protected", ...)

                def strip_bucket(href: str) -> str:
                    # href: https://data.lpdaac.earthdatacloud.nasa.gov/
                    #   lp-prod-protected/path/to/file.tif
                    # store is rooted at the bucket, so the path is
                    # just path/to/file.tif
                    return (
                        urlparse(href).path.lstrip("/").removeprefix("lp-prod-protected/")
                    )

                da = lazycogs.open(
                    "items.parquet", ..., store=store, path_from_href=strip_bucket
                )

        duckdb_client: Optional ``DuckdbClient`` instance.  When
            ``None`` (default), a plain ``DuckdbClient()`` is created,
            which is equivalent to the previous ``rustac.search_sync``
            behaviour.  Pass a custom client to enable features such as
            hive-partitioned datasets::

                import rustac, lazycogs

                client = DuckdbClient(use_hive_partitioning=True)
                da = lazycogs.open(
                    "s3://bucket/stac/",
                    duckdb_client=client,
                    bbox=...,
                    crs=...,
                    resolution=...,
                )

    Returns:
        Lazy ``xr.DataArray`` with dimensions ``(band, time, y, x)``.

    Raises:
        ValueError: If ``href`` is not a ``.parquet`` or ``.geoparquet`` file
            and no *duckdb_client* is provided, if no matching items are
            found, or if ``time_period`` is not a recognised ISO 8601
            duration.

    """
    if duckdb_client is None:
        duckdb_client = DuckdbClient()
        if not href.endswith((".parquet", ".geoparquet")):
            raise ValueError(
                f"href must be a .parquet or .geoparquet file path, "
                f"got: {href!r}. "
                "To search a STAC API, use rustac.search_to() first. "
                "To query a hive-partitioned directory, pass a duckdb_client.",
            )

    # Validate time_period early before any I/O so bad values fail fast.
    grouper = grouper_from_period(time_period)

    dst_crs = CRS.from_user_input(crs)

    epsg_4326 = CRS.from_epsg(4326)
    if dst_crs.equals(epsg_4326):
        bbox_4326 = list(bbox)
    else:
        t = Transformer.from_crs(dst_crs, epsg_4326, always_xy=True)
        xs, ys = t.transform(
            [bbox[0], bbox[2], bbox[0], bbox[2]],
            [bbox[1], bbox[1], bbox[3], bbox[3]],
        )
        bbox_4326 = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

    t0 = time.perf_counter()
    inspection = _inspect_first_item(
        href,
        duckdb_client=duckdb_client,
        bbox=bbox_4326,
        datetime=datetime,
        filter=filter,
        ids=ids,
        bands=bands,
        sortby=sortby,
        store=store,
        path_from_href=path_from_href,
    )
    logger.debug(
        "_inspect_first_item took %.3fs, found %d bands",
        time.perf_counter() - t0,
        len(inspection.bands),
    )

    if not inspection.item_found:
        raise ValueError(
            f"No STAC items matched the query in {href!r} "
            f"(bbox={bbox_4326}, datetime={datetime}).",
        )

    resolved_bands = bands if bands is not None else inspection.bands

    t0 = time.perf_counter()
    filter_strings, time_coords = _build_time_steps(
        href,
        duckdb_client=duckdb_client,
        bbox=bbox_4326,
        datetime=datetime,
        filter=filter,
        ids=ids,
        sortby=sortby,
        temporal_grouper=grouper,
    )
    logger.debug(
        "_build_time_steps took %.3fs, found %d time steps",
        time.perf_counter() - t0,
        len(filter_strings),
    )

    if not filter_strings:
        raise ValueError(
            f"No STAC items matched the query in {href!r} "
            f"(bbox={bbox_4326}, datetime={datetime}).",
        )

    method_cls = mosaic_method if mosaic_method is not None else FirstMethod
    out_dtype, dtype_was_explicit = _resolve_output_dtype(
        [inspection.dtypes[band] for band in resolved_bands],
        dtype=dtype,
        method_cls=method_cls,
    )
    nodata_was_explicit = nodata is not None
    resolved_nodata = (
        nodata
        if nodata_was_explicit
        else _resolve_output_nodata(
            [inspection.nodata_values[band] for band in resolved_bands],
        )
    )

    logger.info(
        "Discovered %d bands and %d time steps.",
        len(resolved_bands),
        len(filter_strings),
    )

    return _build_dataarray(
        parquet_path=href,
        duckdb_client=duckdb_client,
        resolved_bands=resolved_bands,
        filter_strings=filter_strings,
        time_coords=time_coords,
        bbox=bbox,
        bbox_4326=bbox_4326,
        dst_crs=dst_crs,
        resolution=resolution,
        sortby=sortby,
        filter=filter,
        ids=ids,
        nodata=resolved_nodata,
        out_dtype=out_dtype,
        dtype_was_explicit=dtype_was_explicit,
        nodata_was_explicit=nodata_was_explicit,
        method_cls=method_cls,
        chunks=chunks,
        store=store,
        max_concurrent_reads=max_concurrent_reads,
        path_from_href=path_from_href,
    )
