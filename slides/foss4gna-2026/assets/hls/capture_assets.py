"""Capture bounded, current 2025 HLS outputs from the public MAAP index."""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import requests
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from obstore.store import HTTPStore
from pyproj import Transformer
from rustac import DuckdbClient

import lazycogs

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]
ASSET_DIR = Path(__file__).resolve().parent
INDEX = (
    ROOT / "docs/notebooks/data/hls-stac-geoparquet-archive/v2/HLSS30_2.0/**/*.parquet"
)
COLLECTION = "HLSS30_2.0"
DATETIME = "2025-01-01T00:00:00Z/2025-12-31T23:59:59Z"
BANDS = ["B04", "B03", "B02"]
ZONE_BBOX = (-96, 0, -90, 84)
POINT_XY = (576465, 5196159)
WINDOW_HALF_SIZE_M = 2500
TILE_ID_PART = 2
MIN_VALID_DATES = 3


def obtain_store() -> HTTPStore:
    """Use notebook Earthdata authentication without logging secrets."""
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Existing Earthdata environment authentication is unavailable",
        )
    response = requests.post(
        "https://urs.earthdata.nasa.gov/api/users/find_or_create_token",
        auth=(username, password),
        timeout=10,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Earthdata token response did not contain an access token")
    return HTTPStore(
        "https://data.lpdaac.earthdatacloud.nasa.gov",
        client_options={"default_headers": {"Authorization": f"Bearer {token}"}},
    )


def _collect_mgrs_rings(
    items: list[dict],
) -> tuple[list[list[tuple[float, float]]], int]:
    """Collect one year-unioned geometry per MGRS tile from STAC items."""
    rings = []
    seen_tiles = set()
    for item in items:
        item_id = item.get("id", "")
        id_parts = item_id.split(".")
        tile_id = id_parts[TILE_ID_PART] if len(id_parts) > TILE_ID_PART else item_id
        if tile_id in seen_tiles:
            continue
        seen_tiles.add(tile_id)
        geometry = item.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])
        members = [coordinates] if geometry.get("type") == "Polygon" else coordinates
        rings.extend(
            [(float(point[0]), float(point[1])) for point in polygon[0]]
            for polygon in members
            if polygon
        )
    return rings, len(seen_tiles)


def _save_svg(figure: Figure, path: Path) -> None:
    """Save an SVG with trailing whitespace removed for clean diffs."""
    figure.savefig(path)
    svg = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def _write_coverage_map(
    geographic_rings: list[list[tuple[float, float]]],
    point_lonlat: tuple[float, float],
) -> None:
    """Render footprints in a correctly proportioned lon/lat overview."""
    fig, axis = plt.subplots(figsize=(4.5, 4.2), layout="constrained")
    patches = [Polygon(ring, closed=True) for ring in geographic_rings]
    axis.add_collection(
        PatchCollection(
            patches,
            facecolor="#6b9d75",
            edgecolor="#376b43",
            alpha=0.34,
            linewidth=0.45,
        ),
    )
    axis.add_patch(
        Rectangle(
            (-96, 0),
            6,
            84,
            fill=False,
            edgecolor="#35658a",
            linestyle="--",
            linewidth=1.2,
        ),
    )
    axis.scatter(*point_lonlat, color="#c62828", edgecolor="white", s=22, zorder=4)
    axis.text(-95.7, 80, "UTM 15N", color="#35658a", fontsize=8)
    axis.set(
        xlim=(-130, -60),
        ylim=(-5, 85),
        xlabel="Longitude",
        ylabel="Latitude",
        title=(
            "2025 HLS source footprints · Americas\n"
            "302 year-unioned MGRS tiles; not cloud-free pixels"
        ),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(range(-120, -59, 10))
    axis.set_yticks(range(0, 81, 10))
    axis.grid(color="#d9e0e5", linewidth=0.6)
    _save_svg(fig, ASSET_DIR / "coverage.svg")
    plt.close(fig)


def _write_selection_map(
    geographic_rings: list[list[tuple[float, float]]],
    projected_rings: list[list[tuple[float, float]]],
    point_xy: tuple[float, float],
    point_lonlat: tuple[float, float],
) -> None:
    """Render a locator and equal-metre-scale source-footprint inset."""
    fig, (locator, detail) = plt.subplots(
        1,
        2,
        figsize=(9, 5.6),
        gridspec_kw={"width_ratios": (0.7, 1.35)},
        layout="constrained",
    )
    locator.add_collection(
        PatchCollection(
            [Polygon(ring, closed=True) for ring in geographic_rings],
            facecolor="#6b9d75",
            edgecolor="#376b43",
            alpha=0.34,
            linewidth=0.45,
        ),
    )
    locator.add_patch(
        Rectangle(
            (-96, 0),
            6,
            84,
            fill=False,
            edgecolor="#35658a",
            linestyle="--",
            linewidth=1.2,
        ),
    )
    locator.scatter(*point_lonlat, color="#c62828", edgecolor="white", s=22, zorder=4)
    locator.set(
        xlim=(-130, -60),
        ylim=(-5, 85),
        title="Americas locator · UTM zone 15N",
    )
    locator.set_aspect("equal", adjustable="box")
    locator.set_xticks(range(-120, -59, 20))
    locator.set_yticks(range(0, 81, 20))
    locator.grid(color="#d9e0e5", linewidth=0.6)

    x0, y0 = point_xy
    detail.add_collection(
        PatchCollection(
            [Polygon(ring, closed=True) for ring in projected_rings],
            facecolor="#6b9d75",
            edgecolor="#376b43",
            alpha=0.28,
            linewidth=0.7,
        ),
    )
    detail.add_patch(
        Rectangle(
            (x0 - WINDOW_HALF_SIZE_M, y0 - WINDOW_HALF_SIZE_M),
            WINDOW_HALF_SIZE_M * 2,
            WINDOW_HALF_SIZE_M * 2,
            facecolor="#f4b942",
            edgecolor="#a45c00",
            alpha=0.35,
            linewidth=1.5,
            zorder=3,
        ),
    )
    detail.scatter(x0, y0, color="#c62828", edgecolor="white", s=28, zorder=4)
    detail.set(
        xlim=(x0 - 15_000, x0 + 15_000),
        ylim=(y0 - 15_000, y0 + 15_000),
        xlabel="Easting (m), EPSG:32615",
        ylabel="Northing (m)",
        title=(
            "30 km context · equal-metre scale\n"
            "Green footprints · yellow 5 km window · red point\n"
            "Source geometry, not cloud-free pixels"
        ),
    )
    detail.set_aspect("equal", adjustable="box")
    detail.set_xticks(np.arange(x0 - 15_000, x0 + 15_001, 10_000))
    detail.set_yticks(np.arange(y0 - 15_000, y0 + 15_001, 10_000))
    detail.grid(color="#d9e0e5", linewidth=0.6)
    scale_x, scale_y = x0 - 13_000, y0 - 13_500
    detail.plot([scale_x, scale_x + 5_000], [scale_y, scale_y], color="#111827")
    detail.text(scale_x, scale_y + 500, "5 km", fontsize=8)
    fig.suptitle("2025 HLS footprints and selected target window")
    _save_svg(fig, ASSET_DIR / "selection.svg")
    plt.close(fig)


def _write_maps(items: list[dict]) -> int:
    """Write a geographic footprint overview and a metric-scale selection inset."""
    geographic_rings, tile_count = _collect_mgrs_rings(items)
    to_projected = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
    projected_rings = []
    for ring in geographic_rings:
        eastings, northings = to_projected.transform(
            [point[0] for point in ring],
            [point[1] for point in ring],
        )
        projected_rings.append(list(zip(eastings, northings, strict=True)))
    to_geographic = Transformer.from_crs("EPSG:32615", "EPSG:4326", always_xy=True)
    point_lonlat = to_geographic.transform(*POINT_XY)
    _write_coverage_map(geographic_rings, point_lonlat)
    _write_selection_map(geographic_rings, projected_rings, POINT_XY, point_lonlat)
    return tile_count


def capture_current_coverage_map() -> int:
    """Query cached current-year geometry/id fields and refresh both footprint maps."""
    client = DuckdbClient(use_hive_partitioning=True)
    projection_filter = {"op": "=", "args": [{"property": "proj:epsg"}, "32615"]}
    features = client.search(
        str(INDEX),
        bbox=ZONE_BBOX,
        datetime=DATETIME,
        filter=projection_filter,
        exclude=["assets", "links", "stac_extensions", "bbox", "collection"],
    )
    return _write_maps(features)


def capture_point_series(imagery: xr.DataArray) -> int:
    """Compute one year at one pixel and save RGB-band values with nodata gaps."""
    point = imagery.sel(x=POINT_XY[0], y=POINT_XY[1], method="nearest").compute()
    values = np.asarray(point.values)
    nodata = imagery.attrs.get("_FillValue", -9999)
    times = np.asarray(point.time.values).astype("datetime64[D]")
    valid = np.all(values != nodata, axis=0)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < MIN_VALID_DATES:
        raise RuntimeError("Fewer than three valid RGB dates at the selected point")

    fig, axis = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
    for band_index, (band, color) in enumerate(
        zip(BANDS, ("#c62828", "#32824a", "#285ca8"), strict=True),
    ):
        series = values[band_index].astype(float)
        series[series == nodata] = np.nan
        axis.plot(
            times,
            series,
            color=color,
            linewidth=1.0,
            marker="o",
            markersize=2,
            label=band,
        )
    axis.set(title="HLS point RGB through 2025", ylabel="Stored band value (int16 DN)")
    axis.set_xlabel("2025 date; missing observations remain gaps")
    axis.legend(ncol=3, frameon=False)
    fig.savefig(ASSET_DIR / "point-rgb.svg")
    fig.savefig(ASSET_DIR / "point-rgb.png", dpi=120)
    plt.close(fig)
    return len(valid_indices)


def main() -> None:
    """Query the current 2025 index and produce the array and point outputs."""
    if not INDEX.parent.parent.exists():
        raise RuntimeError("Run reproduce.py first to prepare the current year index")

    client = DuckdbClient(use_hive_partitioning=True)
    zone_epsg = "32615"
    projection_filter = {"op": "=", "args": [{"property": "proj:epsg"}, zone_epsg]}
    bbox_4326 = ZONE_BBOX
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
    dst_bbox = transformer.transform_bounds(*bbox_4326)
    sample_item = client.search(
        str(INDEX),
        bbox=bbox_4326,
        datetime=DATETIME,
        filter=projection_filter,
        max_items=1,
    )[0]
    aligned_bbox = lazycogs.align_bbox(
        sample_item["properties"]["proj:transform"],
        dst_bbox,
    )
    store = obtain_store()
    open_args = {
        "href": str(INDEX),
        "bbox": aligned_bbox,
        "crs": "EPSG:32615",
        "resolution": 30,
        "bands": BANDS,
        "filter": projection_filter,
        "datetime": DATETIME,
        "duckdb_client": client,
        "store": store,
    }
    imagery = lazycogs.open(**open_args, time_period="P1D")
    html = imagery._repr_html_()
    css = (Path(xr.__file__).parent / "static/css/style.css").read_text()
    (ASSET_DIR / "array.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<style>{css}</style></head><body>{html}</body></html>",
    )

    point_valid_dates = capture_point_series(imagery)
    nodata = imagery.attrs.get("_FillValue", -9999)
    unique_footprints = capture_current_coverage_map()

    monthly_filter = "proj:epsg = '32615' AND eo:cloud_cover < 50"
    monthly = lazycogs.open(
        href=str(INDEX),
        bbox=aligned_bbox,
        crs="EPSG:32615",
        resolution=30,
        bands=BANDS,
        datetime=DATETIME,
        time_period="P1M",
        filter=monthly_filter,
        sortby="eo:cloud_cover",
        duckdb_client=client,
        store=store,
    )
    monthly_count = monthly.sizes["time"]
    (ASSET_DIR / "snippets.md").write_text(
        "# Verified HLS open snippets\n\n"
        "Both calls were executed against the same current 2025 HLSS30_2.0 "
        "GeoParquet partitions. Setup variables hide auth, archive preparation, "
        "and grid alignment. The monthly open is a separate array definition.\n\n"
        "```python\n"
        "baseline = lazycogs.open(\n"
        "    href=index_href, bbox=zone15_bounds_aligned, crs='EPSG:32615',\n"
        "    resolution=30, bands=['B04', 'B03', 'B02'], time_period='P1D',\n"
        "    datetime='2025-01-01T00:00:00Z/2025-12-31T23:59:59Z',\n"
        "    filter=zone15_filter, duckdb_client=index_client, store=hls_store,\n"
        ")\n\n"
        "monthly = lazycogs.open(\n"
        "    href=index_href, bbox=zone15_bounds_aligned, crs='EPSG:32615',\n"
        "    resolution=30, bands=['B04', 'B03', 'B02'], time_period='P1M',\n"
        "    datetime='2025-01-01T00:00:00Z/2025-12-31T23:59:59Z',\n"
        "    filter=\"proj:epsg = '32615' AND eo:cloud_cover < 50\",\n"
        "    sortby='eo:cloud_cover', duckdb_client=index_client, store=hls_store,\n"
        ")\n```\n\n"
        f"Verified baseline dimensions: {dict(imagery.sizes)}. "
        f"Monthly time coordinates: {monthly_count}. The filter predicate was "
        "run against the current index and sample items expose `eo:cloud_cover`.\n",
    )

    manifest = {
        "status": "READY_WITH_LIMITATIONS",
        "source": {
            "archive": "NASA MAAP HLS STAC GeoParquet archive (Iceberg table)",
            "collection": COLLECTION,
            "iceberg_metadata_url": "https://s3.us-west-2.amazonaws.com/nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/iceberg/metadata/latest.metadata.json",
            "snapshot_id": "69769654033930513",
            "snapshot_manifest_parquet_files": 132,
            "snapshot_manifest_records": 22228347,
            "2025_parquet_files": 12,
            "2025_parquet_compressed_bytes": 665392268,
            "2025_index_records": 2715469,
            "source_command": (
                "uv run --with fastavro python "
                "slides/foss4gna-2026/assets/hls/reproduce.py"
            ),
        },
        "query": {
            "datetime": DATETIME,
            "projection_filter": "proj:epsg = 32615",
            "array_crs": "EPSG:32615",
            "resolution_m": 30,
            "time_period": "P1D",
            "bands": BANDS,
            "dimensions": dict(imagery.sizes),
            "dtype": str(imagery.dtype),
            "nodata": nodata,
            "logical_values": int(np.prod(list(imagery.sizes.values()))),
            "logical_bytes": int(np.prod(list(imagery.sizes.values())))
            * imagery.dtype.itemsize,
            "eo_cloud_cover_sample": sample_item["properties"].get("eo:cloud_cover"),
            "point_xy": POINT_XY,
            "point_valid_dates": point_valid_dates,
            "point_missing_dates": 365 - point_valid_dates,
            "window_size_m": WINDOW_HALF_SIZE_M * 2,
            "coverage_unique_mgrs_tiles": unique_footprints,
            "monthly_time_coordinates": monthly_count,
            "monthly_cloud_filter": monthly_filter,
            "plot_scaling": (
                "RGB montage clips stored int16 values at 2000; raw DNs, "
                "not reflectance-scaled."
            ),
        },
        "source_grid": {
            "sample_item_id": sample_item["id"],
            "projection_epsg": sample_item["properties"].get("proj:epsg"),
            "proj_transform": sample_item["properties"].get("proj:transform"),
            "eo_cloud_cover": sample_item["properties"].get("eo:cloud_cover"),
            "band_metadata": sample_item["assets"]["B04"].get("eo:bands"),
            "raster_bands_scale_unit_metadata": sample_item["assets"]["B04"].get(
                "raster:bands",
            ),
        },
        "authentication": (
            "Existing notebook Earthdata HTTPS flow; no token or password "
            "logged or persisted."
        ),
        "commands": [
            "uv run --with fastavro python "
            "slides/foss4gna-2026/assets/hls/reproduce.py",
            "uv run python slides/foss4gna-2026/assets/hls/capture_assets.py",
            "uv run python slides/foss4gna-2026/assets/hls/capture_nearby_rgb.py",
            "uv run python slides/foss4gna-2026/assets/hls/"
            "capture_assets.py --maps-only",
        ],
        "caveats": [
            "Direct HLS S3 pixel access requires compute in AWS us-west-2; "
            "use HTTPS elsewhere.",
            "eo:cloud_cover is scene-level metadata, not a pixel cloud mask.",
            "Plots display raw stored DNs, not reflectance-scaled values.",
        ],
    }
    (ASSET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info(
        "Captured current 2025 array, %d point dates, %d footprint geometries, "
        "and %d monthly coordinates",
        point_valid_dates,
        unique_footprints,
        monthly_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-only", action="store_true")
    if parser.parse_args().maps_only:
        count = capture_current_coverage_map()
        LOGGER.info("Mapped %d unique MGRS footprints", count)
    else:
        main()
