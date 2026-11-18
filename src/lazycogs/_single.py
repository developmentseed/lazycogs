"""Open a single COG or STAC item at native grid as an xarray DataArray.

Unlike :func:`lazycogs.open`, which mosaics a whole STAC/geoparquet
collection onto a caller-defined output grid, this module reads assets in
place: native CRS, native resolution, native shape, no reprojection.

- :func:`open_cog` reads one Cloud-Optimized GeoTIFF — the obstore-backed
  analogue of ``rioxarray.open_rasterio`` for a single asset.
- :func:`open_item` reads several assets of a single STAC item that share the
  same native grid and stacks them into one ``(band, y, x)`` DataArray whose
  ``band`` coordinate is labelled by asset key.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from async_geotiff import GeoTIFF
from pyproj import CRS
from rasterix import RasterIndex
from xarray import Coordinates, DataArray, Variable
from xarray.core import indexing

from lazycogs._core import (
    _ordered_bands,
    _spatial_coords_with_eager_variables,
    _spatial_ref_dataarray,
)
from lazycogs._executor import run_on_loop
from lazycogs._native_backend import NativeBackendArray
from lazycogs._store import resolve

if TYPE_CHECKING:
    from collections.abc import Callable

    from async_geotiff import Store

__all__ = ["open_cog", "open_cog_async", "open_item", "open_item_async"]


def _cf_attrs(sources: tuple[GeoTIFF, ...]) -> dict[str, object]:
    """Return scalar CF attributes only when every source band agrees."""
    attrs: dict[str, object] = {"grid_mapping": "spatial_ref"}
    nodata = [source.nodata for source in sources for _ in range(source.count)]
    scales = [
        value
        for source in sources
        for value in (source.scales or (1.0,) * source.count)
    ]
    offsets = [
        value
        for source in sources
        for value in (source.offsets or (0.0,) * source.count)
    ]
    for name, values, default in (
        ("_FillValue", nodata, None),
        ("scale_factor", scales, 1.0),
        ("add_offset", offsets, 0.0),
    ):
        first = values[0]
        if (
            first is not None
            and first != default
            and all(
                value == first
                or (value is not None and np.isnan(value) and np.isnan(first))
                for value in values
            )
        ):
            attrs[name] = first
    return attrs


def _build_cog_dataarray(
    sources: tuple[GeoTIFF, ...],
    *,
    band_coord: list[int | str] | list[str] | None = None,
) -> DataArray:
    """Wrap inspected headers in a lazy rioxarray-compatible DataArray."""
    geotiff = sources[0]
    crs = CRS.from_user_input(geotiff.crs)
    index = RasterIndex.from_transform(
        geotiff.transform,
        width=geotiff.width,
        height=geotiff.height,
        x_dim="x",
        y_dim="y",
        crs=crs,
    )
    spatial_coords = _spatial_coords_with_eager_variables(index)
    spatial_ref = _spatial_ref_dataarray(crs, geotiff.transform)
    backend = NativeBackendArray(sources, multiband=band_coord is None)
    bands = band_coord if band_coord is not None else list(range(1, geotiff.count + 1))
    variable = Variable(
        ("band", "y", "x"),
        indexing.LazilyIndexedArray(backend),
        attrs=_cf_attrs(sources),
    )
    return DataArray(
        variable,
        coords=Coordinates({"band": bands, "spatial_ref": spatial_ref})
        | spatial_coords,
    )


async def _open_asset(
    assets: dict[str, Any],
    band: str,
    *,
    store: Store | None,
    path_from_href: Callable[[str], str] | None,
) -> GeoTIFF:
    """Open the COG backing one asset key of a STAC item."""
    href = assets[band].get("href", "")
    if not href:
        raise ValueError(f"Asset {band!r} does not have an href.")
    resolved_store, path = resolve(href, store=store, path_fn=path_from_href)
    return await GeoTIFF.open(path, store=resolved_store)


async def open_cog_async(
    href: str,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open one COG at native resolution as an ``(band, y, x)`` DataArray.

    Async variant of :func:`open_cog` for use inside a running event loop.

    Args:
        href: COG URL (use a ``file://`` URI for local files). With a custom
            store, a store-relative object path is also accepted.
        store: Pre-configured :class:`async_geotiff.Store` for all reads.
        path_from_href: Optional callable ``(href) -> path`` overriding the
            default ``urlparse`` extraction (see :func:`lazycogs.open`).

    Returns:
        DataArray at the COG's native CRS, resolution, and shape — no
        reprojection. Values are not masked or scaled. Shared source metadata
        is exposed as scalar CF attributes.

    """
    resolved_store, path = resolve(href, store=store, path_fn=path_from_href)
    geotiff = await GeoTIFF.open(path, store=resolved_store)
    return _build_cog_dataarray((geotiff,))


def open_cog(
    href: str,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open one COG at native resolution as an ``(band, y, x)`` DataArray.

    Reads a single Cloud-Optimized GeoTIFF in place — native CRS, resolution,
    and shape, no reprojection or mosaicking. Use :func:`lazycogs.open` for a
    reprojected mosaic across a STAC/geoparquet collection, or
    :func:`open_item` to stack several same-grid assets of one STAC item.
    Open fetches metadata without decoding pixels. ``sel`` and ``isel`` do
    not access storage; loading reads all bands in the bounding window.

    Args:
        href: COG URL (use a ``file://`` URI for local files). With a custom
            store, a store-relative object path is also accepted.
        store: Pre-configured :class:`async_geotiff.Store` for all reads.
        path_from_href: Optional callable ``(href) -> path`` overriding the
            default ``urlparse`` extraction (see :func:`lazycogs.open`).

    Returns:
        DataArray at the COG's native CRS, resolution, and shape. Source
        ``nodata`` is set as ``_FillValue`` and any ``scale``/``offset`` as
        ``scale_factor``/``add_offset``.

    """
    return run_on_loop(
        open_cog_async(href, store=store, path_from_href=path_from_href),
    )


def _assets_from_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return the ``assets`` mapping from a STAC item dict."""
    if hasattr(item, "to_dict"):
        item = item.to_dict()
    assets: dict[str, Any] = item.get("assets", {})
    if not assets:
        raise ValueError("STAC item has no assets to open.")
    return assets


async def _inspect_asset_band(
    assets: dict[str, Any],
    band: str,
    *,
    store: Store | None,
    path_from_href: Callable[[str], str] | None,
) -> GeoTIFF:
    """Inspect one single-band asset without reading pixels."""
    geotiff = await _open_asset(
        assets,
        band,
        store=store,
        path_from_href=path_from_href,
    )
    if geotiff.count != 1:
        raise ValueError(
            f"Asset {band!r} is a {geotiff.count}-band COG; open_item stacks "
            "single-band assets. Use lazycogs.open_cog to read a multi-band COG.",
        )
    return geotiff


async def open_item_async(
    item: dict[str, Any],
    bands: list[str] | None = None,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open several same-grid assets of one STAC item as ``(band, y, x)``.

    Async variant of :func:`open_item` for use inside a running event loop.

    Args:
        item: A STAC item as a dict (e.g. a ``rustac`` search result).
        bands: Asset keys to include, in output order. When ``None``, the
            item's preferred data assets are used (role ``"data"`` or media
            type ``image/tiff``), matching :func:`lazycogs.open`.
        store: Pre-configured :class:`async_geotiff.Store` for all reads.
        path_from_href: Optional callable ``(href) -> path`` overriding the
            default ``urlparse`` extraction (see :func:`lazycogs.open`).

    Returns:
        DataArray at the assets' shared native CRS, resolution, and shape, with
        the ``band`` coordinate labelled by asset key. ``nodata``/``scale``/
        ``offset`` are read from each asset file and surfaced as scalar CF
        attrs only when all selected bands agree.

    Raises:
        ValueError: If the item has no assets, an asset key or href is missing,
            an asset is multiband, grids differ, or dtypes cannot be promoted
            without loss.

    """
    assets = _assets_from_item(item)
    resolved_bands = _ordered_bands(assets, bands=bands)
    if not resolved_bands:
        raise ValueError("No assets available to open from the STAC item.")

    sources = await asyncio.gather(
        *[
            _inspect_asset_band(
                assets,
                band,
                store=store,
                path_from_href=path_from_href,
            )
            for band in resolved_bands
        ],
    )

    first = sources[0]
    for band, source in zip(resolved_bands[1:], sources[1:], strict=True):
        differences = []
        if source.crs != first.crs:
            differences.append("CRS")
        if source.transform != first.transform:
            differences.append("transform")
        if (source.height, source.width) != (first.height, first.width):
            differences.append("shape")
        if differences:
            raise ValueError(
                f"Asset {band!r} differs from {resolved_bands[0]!r} in native grid "
                f"fields: {', '.join(differences)}. Use lazycogs.open to mosaic "
                "assets that differ.",
            )
    return _build_cog_dataarray(tuple(sources), band_coord=resolved_bands)


def open_item(
    item: dict[str, Any],
    bands: list[str] | None = None,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open several same-grid assets of one STAC item as ``(band, y, x)``.

    Reads the requested single-band assets of one STAC item at their native
    grid — no reprojection or mosaicking — and stacks them into a single
    DataArray whose ``band`` coordinate is labelled by asset key. All selected
    assets must share the same CRS, affine transform, and shape, and their
    dtypes must promote without loss. Open fetches metadata for each requested
    asset without decoding pixels. ``sel`` and ``isel`` do not access storage;
    loading reads the bounding window from only the selected assets. Use
    :func:`lazycogs.open` for a reprojected mosaic across a whole collection.

    Args:
        item: A STAC item as a dict (e.g. a ``rustac`` search result).
        bands: Asset keys to include, in output order. When ``None``, the
            item's preferred data assets are used (role ``"data"`` or media
            type ``image/tiff``), matching :func:`lazycogs.open`.
        store: Pre-configured :class:`async_geotiff.Store` for all reads.
        path_from_href: Optional callable ``(href) -> path`` overriding the
            default ``urlparse`` extraction (see :func:`lazycogs.open`).

    Returns:
        DataArray at the assets' shared native CRS, resolution, and shape, with
        the ``band`` coordinate labelled by asset key.

    Raises:
        ValueError: If the item has no assets, an asset key or href is missing,
            an asset is multiband, grids differ, or dtypes cannot be promoted
            without loss.

    """
    return run_on_loop(
        open_item_async(
            item,
            bands,
            store=store,
            path_from_href=path_from_href,
        ),
    )
