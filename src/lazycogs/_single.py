"""Open a single COG or STAC item at native grid as an xarray DataArray.

Unlike `lazycogs.open`, which mosaics a whole STAC/geoparquet
collection onto a caller-defined output grid, this module reads assets in
place: native CRS, native resolution, native shape, no reprojection.

- `open_cog` reads one Cloud-Optimized GeoTIFF — the obstore-backed
  analogue of `rioxarray.open_rasterio` for a single asset.
- `open_item` reads several assets of a single STAC item that share the
  same native grid and stacks them into one `(band, y, x)` DataArray whose
  `band` coordinate is labelled by asset key.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from async_geotiff import GeoTIFF
from pyproj import CRS
from rasterix import RasterIndex
from xarray import Coordinates, DataArray, concat

from lazycogs._core import (
    _ordered_bands,
    _spatial_coords_with_eager_variables,
    _spatial_ref_dataarray,
)
from lazycogs._executor import run_on_loop
from lazycogs._store import resolve

if TYPE_CHECKING:
    from collections.abc import Callable

    from async_geotiff import RasterArray, Store

__all__ = ["open_cog", "open_cog_async", "open_item", "open_item_async"]


def _cf_attrs(geotiff: GeoTIFF) -> dict[str, object]:
    """Return CF/rioxarray attrs, attaching only ones that carry meaning."""
    attrs: dict[str, object] = {"grid_mapping": "spatial_ref"}
    if geotiff.nodata is not None:
        attrs["_FillValue"] = geotiff.nodata
    scale = geotiff.scales[0] if geotiff.scales else 1.0
    offset = geotiff.offsets[0] if geotiff.offsets else 0.0
    if scale != 1.0:
        attrs["scale_factor"] = scale
    if offset != 0.0:
        attrs["add_offset"] = offset
    return attrs


def _build_cog_dataarray(
    geotiff: GeoTIFF,
    raster: RasterArray,
    *,
    band_coord: list[int | str] | None = None,
) -> DataArray:
    """Wrap a native-resolution read in a rioxarray-compatible DataArray.

    `band_coord` labels the band dimension; it defaults to 1-based integer
    band indices and is set to the asset key when stacking a STAC item.
    """
    data = raster.data
    crs = CRS.from_user_input(raster.crs)

    index = RasterIndex.from_transform(
        raster.transform,
        width=raster.width,
        height=raster.height,
        x_dim="x",
        y_dim="y",
        crs=crs,
    )
    spatial_coords = _spatial_coords_with_eager_variables(index)
    spatial_ref = _spatial_ref_dataarray(crs, raster.transform)

    bands = band_coord if band_coord is not None else list(range(1, data.shape[0] + 1))
    return DataArray(
        data,
        dims=("band", "y", "x"),
        coords=Coordinates({"band": bands, "spatial_ref": spatial_ref})
        | spatial_coords,
        attrs=_cf_attrs(geotiff),
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
    """Open one COG at native resolution as an `(band, y, x)` DataArray.

    Async variant of `open_cog` for use inside a running event loop.

    Args:
        href: Asset URL or path. When `store` is `None`, an obstore-backed
            store is auto-resolved from the URL root; otherwise only the object
            path is extracted from the HREF.
        store: Pre-configured `async_geotiff.Store` for all reads.
        path_from_href: Optional callable `(href) -> path` overriding the
            default `urlparse` extraction (see `lazycogs.open`).

    Returns:
        DataArray at the COG's native CRS, resolution, and shape — no
        reprojection. Source `nodata` is set as `_FillValue` and any
        `scale`/`offset` as `scale_factor`/`add_offset` so rioxarray's
        `mask_and_scale` decoding applies them.

    """
    resolved_store, path = resolve(href, store=store, path_fn=path_from_href)
    geotiff = await GeoTIFF.open(path, store=resolved_store)
    raster = await geotiff.read()
    return _build_cog_dataarray(geotiff, raster)


def open_cog(
    href: str,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open one COG at native resolution as an `(band, y, x)` DataArray.

    Reads a single Cloud-Optimized GeoTIFF in place — native CRS, resolution,
    and shape, no reprojection or mosaicking. Use `lazycogs.open` for a
    reprojected mosaic across a STAC/geoparquet collection, or
    `open_item` to stack several same-grid assets of one STAC item.

    Args:
        href: Asset URL or path. When `store` is `None`, an obstore-backed
            store is auto-resolved from the URL root; otherwise only the object
            path is extracted from the HREF.
        store: Pre-configured `async_geotiff.Store` for all reads.
        path_from_href: Optional callable `(href) -> path` overriding the
            default `urlparse` extraction (see `lazycogs.open`).

    Returns:
        DataArray at the COG's native CRS, resolution, and shape. Source
        `nodata` is set as `_FillValue` and any `scale`/`offset` as
        `scale_factor`/`add_offset`.

    """
    return run_on_loop(
        open_cog_async(href, store=store, path_from_href=path_from_href),
    )


def _assets_from_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return the `assets` mapping from a STAC item dict."""
    if hasattr(item, "to_dict"):
        item = item.to_dict()
    assets: dict[str, Any] = item.get("assets", {})
    if not assets:
        raise ValueError("STAC item has no assets to open.")
    return assets


async def _read_asset_band(
    assets: dict[str, Any],
    band: str,
    *,
    store: Store | None,
    path_from_href: Callable[[str], str] | None,
) -> DataArray:
    """Open and read one single-band asset as a `(band, y, x)` DataArray."""
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
    raster = await geotiff.read()
    return _build_cog_dataarray(geotiff, raster, band_coord=[band])


async def open_item_async(
    item: dict[str, Any],
    bands: list[str] | None = None,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open several same-grid assets of one STAC item as `(band, y, x)`.

    Async variant of `open_item` for use inside a running event loop.

    Args:
        item: A STAC item as a dict (e.g. a `rustac` search result).
        bands: Asset keys to include, in output order. When `None`, the
            item's preferred data assets are used (role `"data"` or media
            type `image/tiff`), matching `lazycogs.open`.
        store: Pre-configured `async_geotiff.Store` for all reads.
        path_from_href: Optional callable `(href) -> path` overriding the
            default `urlparse` extraction (see `lazycogs.open`).

    Returns:
        DataArray at the assets' shared native CRS, resolution, and shape, with
        the `band` coordinate labelled by asset key. `nodata`/`scale`/
        `offset` are read from each asset file and surfaced as scalar CF
        attrs only when all selected bands agree.

    Raises:
        ValueError: If the item has no assets, a requested band is missing or
            lacks an href, an asset is multi-band, or the selected assets do
            not share one native grid.

    """
    assets = _assets_from_item(item)
    resolved_bands = _ordered_bands(assets, bands=bands)
    if not resolved_bands:
        raise ValueError("No assets available to open from the STAC item.")

    arrays = await asyncio.gather(
        *[
            _read_asset_band(
                assets,
                band,
                store=store,
                path_from_href=path_from_href,
            )
            for band in resolved_bands
        ],
    )

    # join="exact" rejects assets that are not on one native grid; drop_conflicts
    # keeps _FillValue/scale_factor/add_offset only when every band agrees.
    try:
        return concat(arrays, dim="band", join="exact", combine_attrs="drop_conflicts")
    except ValueError as exc:
        raise ValueError(
            "open_item requires every asset to share one native grid (CRS, "
            "resolution, and extent). Use lazycogs.open to mosaic assets that "
            "differ.",
        ) from exc


def open_item(
    item: dict[str, Any],
    bands: list[str] | None = None,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open several same-grid assets of one STAC item as `(band, y, x)`.

    Reads the requested single-band assets of one STAC item at their native
    grid — no reprojection or mosaicking — and stacks them into a single
    DataArray whose `band` coordinate is labelled by asset key. All selected
    assets must share the same native CRS, resolution, and shape. This is the
    multi-band complement to `open_cog`; use `lazycogs.open` for a
    reprojected mosaic across a whole collection.

    Args:
        item: A STAC item as a dict (e.g. a `rustac` search result).
        bands: Asset keys to include, in output order. When `None`, the
            item's preferred data assets are used (role `"data"` or media
            type `image/tiff`), matching `lazycogs.open`.
        store: Pre-configured `async_geotiff.Store` for all reads.
        path_from_href: Optional callable `(href) -> path` overriding the
            default `urlparse` extraction (see `lazycogs.open`).

    Returns:
        DataArray at the assets' shared native CRS, resolution, and shape, with
        the `band` coordinate labelled by asset key.

    Raises:
        ValueError: If the item has no assets, a requested band is missing or
            lacks an href, an asset is multi-band, or the selected assets do
            not share one native grid.

    """
    return run_on_loop(
        open_item_async(
            item,
            bands,
            store=store,
            path_from_href=path_from_href,
        ),
    )
