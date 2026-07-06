"""Small spatial helpers shared across open, backend, and explain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyproj import CRS

from lazycogs._reproject import _get_transformer

if TYPE_CHECKING:
    from affine import Affine


_EPSG_4326 = CRS.from_epsg(4326)


def bbox_to_4326(
    bbox: tuple[float, float, float, float] | list[float],
    crs: CRS,
) -> list[float]:
    """Return ``bbox`` transformed from ``crs`` to EPSG:4326."""
    minx, miny, maxx, maxy = bbox
    if crs.equals(_EPSG_4326):
        return [minx, miny, maxx, maxy]

    xs, ys = _get_transformer(crs, _EPSG_4326).transform(
        [minx, maxx, minx, maxx],
        [maxy, maxy, miny, miny],
    )
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def affine_window_bbox_4326(
    affine: Affine,
    width: int,
    height: int,
    crs: CRS,
) -> list[float]:
    """Return an affine window's bounding box in EPSG:4326."""
    minx = affine.c
    maxy = affine.f
    maxx = minx + width * affine.a
    miny = maxy + height * affine.e
    return bbox_to_4326([minx, miny, maxx, maxy], crs)
