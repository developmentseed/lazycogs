"""Compute output raster grid parameters from bbox, CRS, and resolution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from affine import Affine

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_output_grid(
    bbox: tuple[float, float, float, float],
    resolution: float,
) -> tuple[Affine, int, int]:
    """Compute the output raster grid from a bounding box and resolution.

    The grid is aligned to the bbox corners, with x increasing left-to-right
    and y decreasing top-to-bottom (descending), following the standard
    north-up raster convention.  Label-based slicing with ``xarray.sel`` on
    the ``y`` dimension uses ``slice(north, south)`` (high to low).

    Args:
        bbox: ``(minx, miny, maxx, maxy)`` in the target CRS.
        resolution: Pixel size in CRS units (assumed square).

    Returns:
        A three-tuple ``(transform, width, height)`` where ``transform`` is
        the affine mapping from pixel space to CRS space and ``width`` /
        ``height`` are the grid dimensions.

    """
    minx, miny, maxx, maxy = bbox

    width = max(1, round((maxx - minx) / resolution))
    height = max(1, round((maxy - miny) / resolution))

    # Origin at the top-left corner of the top-left pixel.
    transform = Affine(resolution, 0.0, minx, 0.0, -resolution, maxy)

    return transform, width, height


def align_bbox(
    affine: Affine | Sequence[float],
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Snap a bounding box to the pixel grid defined by an affine transform.

    Expands the bbox outward so that all four edges fall exactly on a grid
    line. Useful for aligning an AOI to the native grid of a COG collection
    (e.g. from a STAC item's ``proj:transform`` property) before calling
    :func:`lazycogs.open`.

    Args:
        affine: Affine transform in row-major order, either 6-element
            ``(pixel_w, 0, x_origin, 0, pixel_h, y_origin)`` or 9-element
            ``(pixel_w, 0, x_origin, 0, pixel_h, y_origin, 0, 0, 1)``.
            Accepts an :class:`affine.Affine` object or the list stored in
            a STAC item's ``proj:transform`` property.
        bbox: ``(minx, miny, maxx, maxy)`` in the same CRS as the transform.

    Returns:
        ``(minx, miny, maxx, maxy)`` snapped to the nearest enclosing grid
        lines.

    """
    if not isinstance(affine, Affine):
        affine = Affine(*affine)
    pixel_w, _, x0, _, pixel_h, y0, _, _, _ = affine
    xmin, ymin, xmax, ymax = bbox

    snapped_xmin = x0 + math.floor((xmin - x0) / pixel_w) * pixel_w
    snapped_xmax = x0 + math.ceil((xmax - x0) / pixel_w) * pixel_w

    abs_h = abs(pixel_h)
    snapped_ymin = y0 + math.floor((ymin - y0) / abs_h) * abs_h
    snapped_ymax = y0 + math.ceil((ymax - y0) / abs_h) * abs_h

    return (snapped_xmin, snapped_ymin, snapped_xmax, snapped_ymax)
