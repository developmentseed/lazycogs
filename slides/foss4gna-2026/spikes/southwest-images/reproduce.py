# ruff: noqa: INP001
"""Render real Southwest Sentinel-2 monthly mosaics and datatake examples."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rustac
from obstore.store import HTTPStore
from PIL import Image, ImageDraw
from pyproj import Transformer

import lazycogs

ROOT = Path(__file__).parent
CACHE = Path(".cache/southwest-images")
INDEX = CACHE / "southwest-2025.parquet"
BOUNDS = (-444000, -609000, 681000, 500000)
CRS = "EPSG:3310"
RESOLUTION = 1200
BANDS = ["red", "green", "blue"]
MONTHS = ["2025-03", "2025-05", "2025-06"]
CLOUD_THRESHOLD = 50


def _http_store(index: str) -> HTTPStore:
    """Build a public HTTP store rooted at the S3 virtual-host endpoint."""
    item = rustac.DuckdbClient().search(index, max_items=1, include=["assets"])[0]
    href = next(
        asset["href"]
        for asset in item["assets"].values()
        if "image/tiff" in asset.get("type", "")
    )
    parsed = urlparse(href)
    return HTTPStore(f"https://{parsed.netloc}")


def _rgb(data: np.ndarray) -> Image.Image:
    """Apply one fixed reflectance display stretch and preserve nodata alpha."""
    valid = np.any(data != 0, axis=0)
    rgb = np.moveaxis(data, 0, -1).astype(np.float32)
    rgb = np.clip(rgb / 5000 * 255, 0, 255).astype(np.uint8)
    rgba = np.dstack((rgb, valid.astype(np.uint8) * 255))
    return Image.fromarray(rgba, mode="RGBA")


def _save_monthly(index: str, store: HTTPStore) -> tuple[dict, dict[str, Image.Image]]:
    """Export same-grid monthly first-valid RGB images."""
    array = lazycogs.open(
        index,
        bbox=BOUNDS,
        crs=CRS,
        resolution=RESOLUTION,
        time_period="P1M",
        bands=BANDS,
        dtype="uint16",
        filter="eo:cloud_cover < 50",
        sortby="eo:cloud_cover",
        store=store,
    )
    images = {}
    info = {
        "dims": dict(array.sizes),
        "crs": CRS,
        "transform": array.attrs["spatial:transform"],
    }
    for month in MONTHS:
        data = array.sel(time=np.datetime64(f"{month}-01")).compute().values
        image = _rgb(data)
        image.save(ROOT / f"mosaic-{month}.png", optimize=True)
        images[month] = image
    return info, images


def _pass_examples(index: str, store: HTTPStore, records: list[dict]) -> list[dict]:
    """Render one cloud-ranked multi-granule datatake group per selected month."""
    output = []
    for month in MONTHS:
        month_items = [
            item
            for item in records
            if item["properties"].get("datetime", "").startswith(month)
            and item["properties"].get("eo:cloud_cover", 100) < CLOUD_THRESHOLD
            and item["properties"].get("s2:datatake_id")
        ]
        counts = Counter(item["properties"]["s2:datatake_id"] for item in month_items)
        for example_number, (datatake, _) in enumerate(counts.most_common(2), 1):
            group = [
                item
                for item in month_items
                if item["properties"]["s2:datatake_id"] == datatake
            ]
            ids = [item["id"] for item in group]
            array = lazycogs.open(
                index,
                bbox=BOUNDS,
                crs=CRS,
                resolution=RESOLUTION,
                time_period="P1D",
                bands=BANDS,
                dtype="uint16",
                ids=ids,
                filter="eo:cloud_cover < 50",
                sortby="eo:cloud_cover",
                store=store,
            )
            data = array.isel(time=0).compute().values
            image = _rgb(data)
            draw = ImageDraw.Draw(image)
            # STAC item geometries use WGS84; project outlines onto the target grid.
            project = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
            for item in group:
                geom = item.get("geometry")
                if not geom:
                    continue
                x0, y0, x1, y1 = BOUNDS
                width, height = image.size
                rings = geom["coordinates"]
                if geom["type"] == "MultiPolygon":
                    rings = [poly[0] for poly in rings]
                elif geom["type"] == "Polygon":
                    rings = [rings[0]]
                else:
                    continue
                for ring in rings:
                    projected = [project.transform(x, y) for x, y, *_ in ring]
                    points = [
                        (
                            (x - x0) / (x1 - x0) * width,
                            (y1 - y) / (y1 - y0) * height,
                        )
                        for x, y in projected
                    ]
                    draw.line(points, fill=(255, 255, 255, 220), width=1)
            path = ROOT / f"datatake-{month}-{example_number}.png"
            image.save(path, optimize=True)
            output.append(
                {
                    "month": month,
                    "datatake_id": datatake,
                    "item_count": len(group),
                    "acquisition_dates": sorted(
                        {item["properties"]["datetime"] for item in group},
                    ),
                    "item_ids": ids,
                    "path": path.name,
                },
            )
    return output


def _previews(images: dict[str, Image.Image], passes: list[dict]) -> None:
    """Create a flat contact sheet of monthly mosaics and selected passes."""
    tiles = [(f"{month} mosaic", image) for month, image in images.items()]
    tiles.extend(
        (
            f"{item['month']} {item['datatake_id']}",
            Image.open(ROOT / item["path"]),
        )
        for item in passes
    )
    tile_width, tile_height = 600, 590
    header = 38
    contact = Image.new("RGB", (tile_width * 3, (tile_height + header) * 3), "#15202b")
    draw = ImageDraw.Draw(contact)
    for index, (label, image) in enumerate(tiles):
        col, row = index % 3, index // 3
        x, y = col * tile_width, row * (tile_height + header)
        tile = image.convert("RGBA")
        tile.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        contact.paste(tile.convert("RGB"), (x, y + header))
        draw.text((x + 10, y + 12), label, fill="white")
    contact.save(ROOT / "contact-sheet.png", optimize=True)


async def main() -> None:
    """Query Earth Search and export bounded HTTP-backed visual assets."""
    CACHE.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    bbox = transformer.transform_bounds(*BOUNDS)
    if not INDEX.exists():
        await rustac.search_to(
            str(INDEX),
            href="https://earth-search.aws.element84.com/v1",
            collections=["sentinel-2-c1-l2a"],
            datetime="2025-03-01/2025-06-30",
            bbox=bbox,
            limit=100,
        )
    client = rustac.DuckdbClient()
    records = list(
        client.search(
            str(INDEX),
            bbox=bbox,
            datetime="2025-03-01/2025-06-30",
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
    store = _http_store(str(INDEX))
    grid, images = _save_monthly(str(INDEX), store)
    passes = _pass_examples(str(INDEX), store, records)
    _previews(images, passes)
    report = {
        "source": "Earth Search STAC API / sentinel-2-c1-l2a",
        "source_items": len(records),
        "source_crs_counts": dict(
            Counter(str(i["properties"].get("proj:epsg")) for i in records),
        ),
        "transport": (
            "obstore HTTPStore at the unchanged S3 virtual-host HTTPS hostname; "
            "object key is the original href path"
        ),
        "target_grid": {
            "crs": CRS,
            "bounds": BOUNDS,
            "resolution_m": RESOLUTION,
            **grid,
        },
        "bands": BANDS,
        "stretch": "linear reflectance DN 0..5000 to RGB 0..255; same all months",
        "mosaic_order": "eo:cloud_cover ascending; first valid pixel",
        "cloud_handling": "scene-level eo:cloud_cover <50; not pixel cloud masking",
        "months": MONTHS,
        "pass_examples": passes,
        "outputs": [p.name for p in ROOT.glob("*.png")],
    }
    (ROOT / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
