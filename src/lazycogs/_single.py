"""Open a single COG at its native grid as an xarray DataArray.

Unlike :func:`lazycogs.open`, which mosaics a whole STAC/geoparquet
collection onto a caller-defined output grid, this module reads one
Cloud-Optimized GeoTIFF in place: native CRS, native resolution, native
shape, no reprojection. It is the obstore-backed analogue of
``rioxarray.open_rasterio`` for a single asset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from async_geotiff import GeoTIFF
from pyproj import CRS
from rasterix import RasterIndex
from xarray import Coordinates, DataArray

from lazycogs._core import _spatial_coords_with_eager_variables
from lazycogs._executor import run_on_loop
from lazycogs._store import resolve

if TYPE_CHECKING:
    from collections.abc import Callable

    from async_geotiff import Store

__all__ = ["open_cog", "open_cog_async"]


def _build_cog_dataarray(geotiff: GeoTIFF, raster: object) -> DataArray:
    """Wrap a native-resolution read in a rioxarray-compatible DataArray.

    Args:
        geotiff: The opened GeoTIFF, used for ``scales``/``offsets``/``nodata``.
        raster: The result of ``geotiff.read()`` with ``data`` (shape
            ``(band, y, x)``), ``transform``, ``crs``, ``width``, ``height``.

    Returns:
        DataArray with dims ``(band, y, x)`` at the COG's native grid.

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

    crs_wkt = crs.to_wkt()
    spatial_ref = DataArray(
        np.array(0),
        attrs={
            "crs_wkt": crs_wkt,
            "spatial_ref": crs_wkt,
            "GeoTransform": " ".join(str(v) for v in raster.transform.to_gdal()),
        },
    )

    attrs: dict[str, object] = {"grid_mapping": "spatial_ref"}
    if geotiff.nodata is not None:
        attrs["_FillValue"] = geotiff.nodata
    scale = geotiff.scales[0] if geotiff.scales else 1.0
    offset = geotiff.offsets[0] if geotiff.offsets else 0.0
    if scale != 1.0:
        attrs["scale_factor"] = scale
    if offset != 0.0:
        attrs["add_offset"] = offset

    bands = list(range(1, data.shape[0] + 1))
    return DataArray(
        data,
        dims=("band", "y", "x"),
        coords=Coordinates({"band": bands, "spatial_ref": spatial_ref})
        | spatial_coords,
        attrs=attrs,
    )


async def open_cog_async(
    href: str,
    *,
    store: Store | None = None,
    path_from_href: Callable[[str], str] | None = None,
) -> DataArray:
    """Open one COG at native resolution as an ``(band, y, x)`` DataArray.

    Async variant of :func:`open_cog` for use inside a running event loop.

    Args:
        href: Asset URL or path. When ``store`` is ``None``, an obstore-backed
            store is auto-resolved from the URL root; otherwise only the object
            path is extracted from the HREF.
        store: Pre-configured :class:`async_geotiff.Store` for all reads.
        path_from_href: Optional callable ``(href) -> path`` overriding the
            default ``urlparse`` extraction (see :func:`lazycogs.open`).

    Returns:
        DataArray at the COG's native CRS, resolution, and shape — no
        reprojection. Source ``nodata`` is set as ``_FillValue`` and any
        ``scale``/``offset`` as ``scale_factor``/``add_offset`` so rioxarray's
        ``mask_and_scale`` decoding applies them.

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
    """Open one COG at native resolution as an ``(band, y, x)`` DataArray.

    Reads a single Cloud-Optimized GeoTIFF in place — native CRS, resolution,
    and shape, no reprojection or mosaicking. Use :func:`lazycogs.open` for a
    reprojected mosaic across a STAC/geoparquet collection.

    Args:
        href: Asset URL or path. When ``store`` is ``None``, an obstore-backed
            store is auto-resolved from the URL root; otherwise only the object
            path is extracted from the HREF.
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
