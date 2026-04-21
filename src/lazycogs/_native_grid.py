"""Discover the native CRS and pixel grid of a STAC COG collection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from affine import Affine
from pyproj import CRS, Transformer
from rustac import DuckdbClient

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class NativeGrid:
    """Native CRS, resolution, and pixel-snapped bbox for a STAC collection.

    Produced by :func:`lazycogs.native_grid` and consumed directly by
    :func:`lazycogs.open`::

        grid = lazycogs.native_grid("items.parquet", bbox=[-108.5, 37.5, -107.5, 38.5])
        da = lazycogs.open(
            "items.parquet",
            bbox=grid.bbox,
            crs=grid.crs,
            resolution=grid.resolution,
        )

    Attributes:
        crs: Authority string for the native CRS (e.g. ``"EPSG:32612"``).
        resolution: Native pixel size in CRS units.
        bbox: ``(minx, miny, maxx, maxy)`` in the native CRS, snapped outward
            to the nearest pixel-grid boundary so that :func:`lazycogs.open`
            can use the no-reprojection fast path.
    """

    crs: str
    resolution: float
    bbox: tuple[float, float, float, float]


def _epsg_from_item(item: dict[str, Any]) -> int | None:
    """Return the EPSG code from an item's properties, or ``None``."""
    return item.get("properties", {}).get("proj:epsg")


def _transform_from_asset(asset: dict[str, Any], band: str, item_id: str) -> Affine:
    """Extract the ``Affine`` transform from an asset's ``proj:transform`` field.

    Raises:
        ValueError: If ``proj:transform`` is absent from the asset.
    """
    pt = asset.get("proj:transform")
    if pt is None:
        raise ValueError(
            f"Asset {band!r} in item {item_id!r} is missing 'proj:transform'. "
            "Ensure the collection uses the STAC projection extension "
            "(https://github.com/stac-extensions/projection)."
        )
    a, b, c, d, e, f = pt[:6]
    return Affine(a, b, c, d, e, f)


def _snap_bbox(
    proj_minx: float,
    proj_miny: float,
    proj_maxx: float,
    proj_maxy: float,
    grid_transform: Affine,
) -> tuple[float, float, float, float]:
    """Snap a projected bbox outward to the nearest pixel-grid boundary.

    Args:
        proj_minx: Minimum x coordinate in the native CRS.
        proj_miny: Minimum y coordinate in the native CRS.
        proj_maxx: Maximum x coordinate in the native CRS.
        proj_maxy: Maximum y coordinate in the native CRS.
        grid_transform: Affine transform whose origin and pixel size define the
            native pixel grid.

    Returns:
        ``(minx, miny, maxx, maxy)`` snapped to pixel boundaries, guaranteed
        to contain the input bbox.
    """
    ox = grid_transform.c
    px = grid_transform.a  # positive pixel width
    oy = grid_transform.f
    py = grid_transform.e  # negative pixel height

    snapped_minx = ox + math.floor((proj_minx - ox) / px) * px
    snapped_maxx = ox + math.ceil((proj_maxx - ox) / px) * px
    # py is negative, so floor/ceil roles are swapped for y.
    snapped_maxy = oy + math.floor((proj_maxy - oy) / py) * py
    snapped_miny = oy + math.ceil((proj_miny - oy) / py) * py

    return snapped_minx, snapped_miny, snapped_maxx, snapped_maxy


def native_grid(
    parquet: str,
    bbox: tuple[float, float, float, float],
    bands: list[str] | None = None,
    *,
    datetime: str | None = None,
    duckdb_client: DuckdbClient | None = None,
) -> NativeGrid:
    """Discover the native CRS and resolution for a STAC COG collection.

    Reads ``proj:epsg`` and ``proj:transform`` from STAC item asset metadata
    (populated by the `STAC projection extension
    <https://github.com/stac-extensions/projection>`_) without opening any COG
    files.  Returns the shared native CRS and pixel size, plus a
    pixel-grid-aligned version of *bbox* that enables the no-reprojection fast
    path in :func:`lazycogs.open`.

    Example::

        grid = lazycogs.native_grid(
            "sentinel2.parquet",
            bbox=[-108.5, 37.5, -107.5, 38.5],
            bands=["red"],
        )
        da = lazycogs.open(
            "sentinel2.parquet",
            bbox=grid.bbox,
            crs=grid.crs,
            resolution=grid.resolution,
        )

    Args:
        parquet: Path to a geoparquet file (``.parquet`` or ``.geoparquet``)
            or a hive-partitioned parquet directory when *duckdb_client* is
            provided.
        bbox: ``(minx, miny, maxx, maxy)`` in EPSG:4326 describing the area of
            interest.  The returned :attr:`NativeGrid.bbox` covers at least
            this area.
        bands: Asset keys to inspect.  When ``None``, all data assets from the
            first matching item are used.  All requested bands must share the
            same native resolution; pass a single band name when bands have
            different native resolutions (e.g. Sentinel-2 10 m red vs 20 m
            NIR).
        datetime: RFC 3339 datetime or interval (e.g. ``"2024-06-01/2024-06-30"``)
            to filter items before inspecting their metadata.
        duckdb_client: Optional ``DuckdbClient`` instance for hive-partitioned
            datasets.  Defaults to a plain ``DuckdbClient()``.

    Returns:
        :class:`NativeGrid` containing the shared ``crs``, native
        ``resolution``, and a pixel-grid-aligned ``bbox`` in the native CRS.

    Raises:
        ValueError: If no matching items are found; if ``bands`` are absent
            from the asset metadata; if ``proj:transform`` is missing from any
            asset; if items do not share a common CRS (e.g. a bbox spanning two
            UTM zones); or if the requested bands have different native
            resolutions.
    """
    if duckdb_client is None:
        duckdb_client = DuckdbClient()

    items = duckdb_client.search(parquet, bbox=list(bbox), datetime=datetime)
    if not items:
        raise ValueError(f"No STAC items found in {parquet!r} for bbox={bbox}")

    # Determine which bands to inspect from the first item when not specified.
    first_assets: dict[str, Any] = items[0].get("assets", {})
    if bands is None:
        bands = [
            k
            for k, v in first_assets.items()
            if "data" in v.get("roles", []) or "image/tiff" in v.get("type", "")
        ] or list(first_assets)

    crs_values: set[int] = set()
    # One representative transform per band (pixel size and grid origin).
    band_transforms: dict[str, Affine] = {}

    for item in items:
        epsg = _epsg_from_item(item)
        if epsg is not None:
            crs_values.add(epsg)

        item_id: str = item.get("id", "<unknown>")
        assets = item.get("assets", {})
        for band in bands:
            asset = assets.get(band)
            if asset is None:
                continue
            transform = _transform_from_asset(asset, band, item_id)
            if band in band_transforms:
                existing = band_transforms[band]
                if abs(existing.a) != abs(transform.a) or abs(existing.e) != abs(
                    transform.e
                ):
                    raise ValueError(
                        f"Band {band!r} has inconsistent native pixel size across "
                        f"items ({abs(existing.a)} vs {abs(transform.a)} CRS units). "
                        "Narrow the bbox to a single tile."
                    )
            else:
                band_transforms[band] = transform

    missing = [b for b in bands if b not in band_transforms]
    if missing:
        raise ValueError(f"Band(s) {missing} not found in any matching item's assets.")

    if not crs_values:
        raise ValueError(
            "No 'proj:epsg' found in item properties. Ensure the collection "
            "uses the STAC projection extension."
        )
    if len(crs_values) > 1:
        raise ValueError(
            f"Items span multiple CRSs: {crs_values}. "
            "Narrow the bbox to a single UTM zone or tile."
        )

    resolutions = {band: abs(t.a) for band, t in band_transforms.items()}
    unique_res = set(resolutions.values())
    if len(unique_res) > 1:
        raise ValueError(
            f"Requested bands have different native resolutions: {resolutions}. "
            "Specify a single band name to get a unique native resolution."
        )

    epsg = next(iter(crs_values))
    resolution = unique_res.pop()
    grid_transform = next(iter(band_transforms.values()))

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    minx_4326, miny_4326, maxx_4326, maxy_4326 = bbox
    xs, ys = transformer.transform(
        [minx_4326, maxx_4326, minx_4326, maxx_4326],
        [miny_4326, miny_4326, maxy_4326, maxy_4326],
    )
    proj_minx, proj_maxx = min(xs), max(xs)
    proj_miny, proj_maxy = min(ys), max(ys)

    snapped_bbox = _snap_bbox(
        proj_minx, proj_miny, proj_maxx, proj_maxy, grid_transform
    )

    return NativeGrid(crs=f"EPSG:{epsg}", resolution=resolution, bbox=snapped_bbox)
