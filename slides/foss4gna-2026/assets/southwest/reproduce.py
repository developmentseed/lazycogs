# ruff: noqa: INP001
"""Build final Southwest seasonal RGB, pass, repr, and source-grid evidence."""

from __future__ import annotations

import asyncio
import calendar
import gc
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rustac
from async_geotiff import GeoTIFF
from obstore.store import HTTPStore
from PIL import Image, ImageDraw
from pyproj import Transformer

import lazycogs
from lazycogs._chunk_reader import _select_overview

ROOT = Path(__file__).parent
CACHE = Path(".cache/southwest-images")
INDEX = CACHE / "southwest-2025-full.parquet"
BOUNDS = (-444000, -609000, 681000, 500000)
CRS = "EPSG:3310"
RESOLUTION = 300
DISPLAY_RESOLUTION = 1200
MONTHS = ["2025-03", "2025-07", "2025-11"]
BANDS = ["red", "green", "blue"]
CLOUD_FILTER = "eo:cloud_cover < 50"
CLOUD_THRESHOLD = 50


def _store(index: str) -> HTTPStore:
    """Create an anonymous HTTPS store for the public Earth Search assets."""
    item = rustac.DuckdbClient().search(index, max_items=1, include=["assets"])[0]
    href = next(
        asset["href"]
        for asset in item["assets"].values()
        if "image/tiff" in asset.get("type", "")
    )
    parsed = urlparse(href)
    return HTTPStore(f"https://{parsed.netloc}")


def _rgb(data: np.ndarray) -> Image.Image:
    """Apply a common display stretch and transparent alpha for nodata."""
    valid = np.any(data != 0, axis=0)
    rgb = np.moveaxis(data, 0, -1).astype(np.float32)
    rgb = np.clip(rgb / 5000 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack((rgb, valid.astype(np.uint8) * 255)), mode="RGBA")


def _items(index: str) -> list[dict]:
    """Read only catalog fields required for selected months and footprints."""
    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    bbox = transformer.transform_bounds(*BOUNDS)
    client = rustac.DuckdbClient()
    items = []
    for month in MONTHS:
        year, month_number = (int(value) for value in month.split("-"))
        start = f"{month}-01"
        end = f"{month}-{calendar.monthrange(year, month_number)[1]}"
        items.extend(
            client.search(
                index,
                bbox=bbox,
                datetime=f"{start}/{end}",
                include=[
                    "id",
                    "datetime",
                    "eo:cloud_cover",
                    "s2:datatake_id",
                    "proj:epsg",
                    "geometry",
                ],
            ),
        )
    return items


def _monthly_array(
    index: str,
    store: HTTPStore,
) -> tuple[object, dict[str, Image.Image]]:
    """Capture the 300 m repr and sequentially render 1200 m previews."""
    array = lazycogs.open(
        index,
        bbox=BOUNDS,
        crs=CRS,
        resolution=RESOLUTION,
        time_period="P1M",
        bands=BANDS,
        dtype="uint16",
        filter=CLOUD_FILTER,
        sortby="eo:cloud_cover",
        max_concurrent_reads=2,
        store=store,
    )
    selected = array.sel(time=[np.datetime64(f"{month}-01") for month in MONTHS])
    # Preserve Xarray's actual self-contained HTML repr, not a hand-built summary.
    repr_html = selected._repr_html_()
    (ROOT / "array.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Southwest monthly DataArray</title></head>"
        f"<body>{repr_html}</body></html>",
        encoding="utf-8",
    )
    preview = lazycogs.open(
        index,
        bbox=BOUNDS,
        crs=CRS,
        resolution=DISPLAY_RESOLUTION,
        time_period="P1M",
        bands=BANDS,
        dtype="uint16",
        filter=CLOUD_FILTER,
        sortby="eo:cloud_cover",
        max_concurrent_reads=2,
        store=store,
    ).sel(time=[np.datetime64(f"{month}-01") for month in MONTHS])
    images = {}
    for month in MONTHS:
        image = _rgb(preview.sel(time=np.datetime64(f"{month}-01")).compute().values)
        image.save(ROOT / f"mosaic-{month}.png", optimize=True)
        images[month] = image
        del image
        gc.collect()
    return selected, images


def _pass_groups(items: list[dict]) -> list[dict]:
    """Choose two qualifying multi-granule datatakes per selected month."""
    selected = []
    for month in MONTHS:
        month_items = [
            item
            for item in items
            if item["properties"].get("datetime", "").startswith(month)
            and item["properties"].get("eo:cloud_cover", 100) < CLOUD_THRESHOLD
            and item["properties"].get("s2:datatake_id")
        ]
        groups = Counter(item["properties"]["s2:datatake_id"] for item in month_items)
        for datatake, _ in groups.most_common(2):
            group = [
                item
                for item in month_items
                if item["properties"]["s2:datatake_id"] == datatake
            ]
            selected.append(
                {
                    "month": month,
                    "datatake_id": datatake,
                    "item_count": len(group),
                    "items": group,
                },
            )
    return selected


async def _pass_images(index: str, store: HTTPStore, groups: list[dict]) -> list[dict]:
    """Read selected acquisition groups on the common 300 m target grid."""
    results = []
    month_counts: Counter[str] = Counter()
    for group in groups:
        month_counts[group["month"]] += 1
        group_number = month_counts[group["month"]]
        array = lazycogs.open(
            index,
            bbox=BOUNDS,
            crs=CRS,
            resolution=DISPLAY_RESOLUTION,
            time_period="P1D",
            bands=BANDS,
            dtype="uint16",
            ids=[item["id"] for item in group["items"]],
            filter=CLOUD_FILTER,
            sortby="eo:cloud_cover",
            max_concurrent_reads=2,
            store=store,
        )
        data = array.isel(time=0).compute().values
        image = _rgb(data)
        draw = ImageDraw.Draw(image)
        project = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        for item in group["items"]:
            geometry = item.get("geometry")
            if not geometry:
                continue
            rings = geometry["coordinates"]
            if geometry["type"] == "Polygon":
                rings = [rings[0]]
            elif geometry["type"] == "MultiPolygon":
                rings = [polygon[0] for polygon in rings]
            else:
                continue
            for ring in rings:
                projected = [project.transform(x, y) for x, y, *_ in ring]
                points = [
                    (
                        (x - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]) * image.width,
                        (BOUNDS[3] - y) / (BOUNDS[3] - BOUNDS[1]) * image.height,
                    )
                    for x, y in projected
                ]
                draw.line(points, fill=(255, 255, 255, 220), width=2)
        path = ROOT / f"datatake-{group['month']}-{group_number}.png"
        image.save(path, optimize=True)
        del data, image, array
        gc.collect()
        results.append(
            {
                "month": group["month"],
                "datatake_id": group["datatake_id"],
                "item_count": group["item_count"],
                "item_ids": [item["id"] for item in group["items"]],
                "acquisition_datetimes": sorted(
                    {item["properties"]["datetime"] for item in group["items"]},
                ),
                "output": path.name,
            },
        )
    return results


def _footprint_map(groups: list[dict]) -> dict[str, int]:
    """Draw exactly the selected pass-item footprints, colored by source EPSG."""
    items = [item for group in groups for item in group["items"]]
    counts = Counter(str(item["properties"].get("proj:epsg")) for item in items)
    colors = dict(
        zip(
            sorted(counts),
            ("#d94841", "#2b8cbe", "#31a354", "#756bb1", "#e6ab02"),
            strict=False,
        ),
    )
    project = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    width, height = 1000, 900
    paths = []
    for item in items:
        geometry = item.get("geometry")
        if not geometry:
            continue
        rings = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            rings = [rings[0]]
        elif geometry["type"] == "MultiPolygon":
            rings = [polygon[0] for polygon in rings]
        else:
            continue
        for ring in rings:
            projected = [project.transform(x, y) for x, y, *_ in ring]
            points = [
                (
                    (x - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]) * width,
                    (BOUNDS[3] - y) / (BOUNDS[3] - BOUNDS[1]) * height,
                )
                for x, y in projected
            ]
            d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points) + " Z"
            color = colors[str(item["properties"].get("proj:epsg"))]
            paths.append(
                f'<path d="{d}" fill="{color}" fill-opacity="0.10" '
                f'stroke="{color}" stroke-width="0.8"/>',
            )
    legend = "".join(
        f'<text x="1020" y="{30 + n * 26}" font-size="16" '
        f'fill="{color}">EPSG:{code} — {counts[code]} granules</text>'
        for n, (code, color) in enumerate(sorted(colors.items()))
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="940" '
        'viewBox="0 0 1450 940"><rect width="100%" height="100%" fill="#fff"/>'
        f"<g>{''.join(paths)}</g>{legend}</svg>"
    )
    (ROOT / "footprints.svg").write_text(svg, encoding="utf-8")
    return dict(counts)


async def _overview_evidence(
    index: str,
    groups: list[dict],
) -> list[dict]:
    """Inspect selected COG headers and record the 300 m overview choice."""
    sampled: dict[tuple[str, str], dict] = {}
    for group in groups:
        for item in group["items"]:
            crs_code = str(item["properties"].get("proj:epsg"))
            sampled.setdefault((group["month"], crs_code), item)
    asset_records = rustac.DuckdbClient().search(
        index,
        ids=[item["id"] for item in sampled.values()],
        include=["id", "assets"],
    )
    assets_by_id = {item["id"]: item["assets"] for item in asset_records}
    for key, item in list(sampled.items()):
        sampled[key] = item | {"assets": assets_by_id[item["id"]]}
    evidence = []
    for (month, crs_code), item in sorted(sampled.items()):
        assets = item.get("assets", {})
        red = next(
            asset
            for asset in assets.values()
            if asset.get("eo:bands", [{}])[0].get("common_name") == "red"
        )
        href = red["href"]
        parsed = urlparse(href)
        source_store = HTTPStore(f"https://{parsed.netloc}")
        geotiff = await GeoTIFF.open(parsed.path.lstrip("/"), store=source_store)
        source_crs = geotiff.crs
        target_to_source = Transformer.from_crs(CRS, source_crs, always_xy=True)
        cx = (BOUNDS[0] + BOUNDS[2]) / 2
        cy = (BOUNDS[1] + BOUNDS[3]) / 2
        x0, _ = target_to_source.transform(cx, cy)
        x1, _ = target_to_source.transform(cx + RESOLUTION, cy)
        native_target_resolution = abs(x1 - x0)
        overview = _select_overview(geotiff, native_target_resolution)
        evidence.append(
            {
                "month": month,
                "item_id": item["id"],
                "source_epsg": crs_code,
                "band": "red",
                "source_pixel_size_m": abs(geotiff.transform.a),
                "overview_resolutions_m": [
                    abs(level.transform.a) for level in geotiff.overviews
                ],
                "target_resolution_in_source_crs_m": native_target_resolution,
                "selected_overview_resolution_m": (
                    None if overview is None else abs(overview.transform.a)
                ),
                "decision": "full-resolution fallback"
                if overview is None
                else "coarsest overview <= target source-CRS resolution",
            },
        )
    return evidence


def _contact_sheet() -> None:
    """Create a flat contact sheet containing every final raster image."""
    paths = sorted(ROOT.glob("mosaic-*.png")) + sorted(ROOT.glob("datatake-*.png"))
    tile_width, tile_height, header = 500, 500, 34
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + header)),
        "#17202a",
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + header)
        sheet.paste(image.convert("RGB"), (x, y + header))
        draw.text((x + 8, y + 10), path.stem, fill="white")
    sheet.save(ROOT / "contact-sheet.png", optimize=True)


async def main() -> None:
    """Search one example year and export actual 300 m display assets/evidence."""
    CACHE.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    bbox = transformer.transform_bounds(*BOUNDS)
    if not INDEX.exists():
        await rustac.search_to(
            str(INDEX),
            href="https://earth-search.aws.element84.com/v1",
            collections=["sentinel-2-c1-l2a"],
            datetime="2025-01-01/2025-12-31",
            bbox=bbox,
            limit=100,
        )
    items = _items(str(INDEX))
    store = _store(str(INDEX))
    selected, _images = _monthly_array(str(INDEX), store)
    groups = _pass_groups(items)
    passes = await _pass_images(str(INDEX), store, groups)
    footprint_counts = _footprint_map(groups)
    overviews = await _overview_evidence(str(INDEX), groups)
    _contact_sheet()
    results = {
        "source": "Earth Search sentinel-2-c1-l2a",
        "full_year_index": str(INDEX),
        "display_resolution_m": DISPLAY_RESOLUTION,
        "index_datetime": "2025-01-01/2025-12-31",
        "queried_months": MONTHS,
        "item_count": len(items),
        "source_crs_counts": dict(
            Counter(str(item["properties"].get("proj:epsg")) for item in items),
        ),
        "bounds_epsg3310": BOUNDS,
        "target_grid": {
            "crs": CRS,
            "resolution_m": RESOLUTION,
            "display_preview_resolution_m": DISPLAY_RESOLUTION,
            "dimensions": dict(selected.sizes),
            "transform": selected.attrs["spatial:transform"],
        },
        "months": MONTHS,
        "pass_groups": passes,
        "footprint_map_subset_counts": footprint_counts,
        "overview_evidence": overviews,
        "runtime_versions": {
            name: version(package)
            for name, package in [
                ("lazycogs", "lazycogs"),
                ("rustac", "rustac"),
                ("obstore", "obstore"),
                ("async-geotiff", "async-geotiff"),
                ("xarray", "xarray"),
                ("numpy", "numpy"),
                ("Pillow", "Pillow"),
            ]
        },
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
