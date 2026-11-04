"""Capture bounded, current 2025 HLS outputs from the public MAAP index."""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from matplotlib import pyplot as plt
from obstore.store import HTTPStore
from pyproj import Transformer
from rustac import DuckdbClient

import lazycogs

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


def _transform_ring(
    ring: list[list[float]],
    transformer: Transformer,
) -> list[tuple[float, float]]:
    """Transform one GeoJSON polygon ring into target-grid coordinates."""
    x_values, y_values = transformer.transform(
        [coordinate[0] for coordinate in ring],
        [coordinate[1] for coordinate in ring],
    )
    return list(zip(x_values, y_values, strict=True))


def _write_maps(
    items: list[dict],
    transformer: Transformer,
    bounds: tuple[float, ...],
) -> int:
    """Write current item-footprint coverage plus the demonstrated selection."""
    polygons = {}
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
        for polygon in members:
            if not polygon:
                continue
            ring = _transform_ring(polygon[0], transformer)
            key = tuple((round(x, 1), round(y, 1)) for x, y in ring)
            polygons[key] = ring

    left, top, width, height = 58, 54, 334, 232
    xmin, ymin, xmax, ymax = bounds
    sx = width / (xmax - xmin)
    sy = height / (ymax - ymin)
    points = []
    for ring in polygons.values():
        xy = " ".join(
            f"{left + (x - xmin) * sx:.1f},{top + (ymax - y) * sy:.1f}" for x, y in ring
        )
        points.append(f'<polygon points="{xy}"/>')

    px, py = POINT_XY
    wx0, wx1 = px - WINDOW_HALF_SIZE_M, px + WINDOW_HALF_SIZE_M
    wy0, wy1 = py - WINDOW_HALF_SIZE_M, py + WINDOW_HALF_SIZE_M
    point_svg = (left + (px - xmin) * sx, top + (ymax - py) * sy)
    window_svg = (
        left + (wx0 - xmin) * sx,
        top + (ymax - wy1) * sy,
        (wx1 - wx0) * sx,
        (wy1 - wy0) * sy,
    )
    body = "\n".join(points)
    svg = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="450" height="350"',
            'viewBox="0 0 450 350">',
            '<rect width="450" height="350" fill="#fff"/>',
            '<text x="225" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="15">'
            "2025 HLS MGRS tile coverage - UTM 15N</text>",
            f'<g fill="#6b9d75" fill-opacity=".38" '
            f'stroke="#376b43" stroke-width=".45">{body}</g>',
            f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
            'fill="none" stroke="#334155" stroke-width="1.2"/>',
            f'<rect x="{window_svg[0]:.1f}" y="{window_svg[1]:.1f}" '
            f'width="{window_svg[2]:.1f}" height="{window_svg[3]:.1f}" '
            'fill="#f4b942" fill-opacity=".42" stroke="#a45c00" '
            'stroke-width="1.5"/>',
            f'<circle cx="{point_svg[0]:.1f}" cy="{point_svg[1]:.1f}" '
            'r="4.5" fill="#c62828" stroke="white" stroke-width="1.5"/>',
            f'<text x="{left}" y="304" font-family="sans-serif" '
            'font-size="12">Year-unioned MGRS footprints · point · 5 km window</text>',
            f'<text x="{left}" y="323" font-family="sans-serif" '
            'font-size="11" fill="#475569">Tile coverage, not cloud-free '
            "pixel coverage</text>",
            "</svg>",
        ],
    )
    (ASSET_DIR / "coverage.svg").write_text(svg)
    (ASSET_DIR / "selection.svg").write_text(svg)
    return len(polygons)


def capture_current_coverage_map() -> int:
    """Query current 2025 geometry/id fields only and refresh both footprint maps."""
    client = DuckdbClient(use_hive_partitioning=True)
    bbox = ZONE_BBOX
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
    dst_bbox = transformer.transform_bounds(*bbox)
    projection_filter = {"op": "=", "args": [{"property": "proj:epsg"}, "32615"]}
    sample = client.search(
        str(INDEX),
        bbox=bbox,
        datetime=DATETIME,
        filter=projection_filter,
        max_items=1,
        exclude=["assets", "links", "stac_extensions", "bbox", "collection"],
    )[0]
    bounds = lazycogs.align_bbox(sample["properties"]["proj:transform"], dst_bbox)
    features = client.search(
        str(INDEX),
        bbox=bbox,
        datetime=DATETIME,
        filter=projection_filter,
        exclude=["assets", "links", "stac_extensions", "bbox", "collection"],
    )
    return _write_maps(features, transformer, bounds)


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
        axis.plot(times, series, color=color, linewidth=1.0, label=band)
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
        LOGGER.info("Mapped %d current item geometries", count)
    else:
        main()
