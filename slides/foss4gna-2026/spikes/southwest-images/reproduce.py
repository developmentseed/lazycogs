# ruff: noqa: INP001
"""Create low-resolution Southwest Sentinel-2 monthly mosaic review assets."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import rustac
from PIL import Image, ImageDraw
from pyproj import Transformer

import lazycogs

ROOT = Path(__file__).parent
CACHE = Path(".cache/southwest-images")
INDEX = CACHE / "southwest-2025.parquet"
BOUNDS = (-444000, -609000, 681000, 500000)
CRS = "EPSG:3310"
# Coarsened from the notebook's 300 m grid to bound this review-only read.
RESOLUTION = 1200
MONTHS = ["2025-03", "2025-05", "2025-06"]
BANDS = ["red", "green", "blue"]


async def main() -> None:
    """Search the public catalog and export selected monthly RGB images."""
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
    queried = client.search(
        str(INDEX),
        bbox=bbox,
        datetime="2025-03-01/2025-06-30",
        include=[
            "id",
            "datetime",
            "eo:cloud_cover",
            "s2:datatake_id",
            "s2:granule_id",
            "proj:epsg",
            "proj:wkt2",
            "geometry",
        ],
    )
    records = list(queried)
    images: dict[str, Image.Image] = {}
    reports = {}
    for month in MONTHS:
        start = f"{month}-01"
        # Time steps are calendar months; arrays share the notebook's exact extent.
        da = lazycogs.open(
            str(INDEX),
            bbox=BOUNDS,
            crs=CRS,
            resolution=RESOLUTION,
            time_period="P1M",
            bands=BANDS,
            dtype="uint16",
            filter="eo:cloud_cover < 50",
            sortby="eo:cloud_cover",
        )
        date = np.datetime64(start)
        selected = da.sel(time=date).compute().values
        # Sentinel-2 L2A reflectance is encoded in integer scale. A fixed linear
        # stretch is applied identically to all three dates for comparison.
        rgb = np.moveaxis(selected, 0, -1).astype(np.float32)
        rgb = np.clip(rgb / 2500 * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        images[month] = image
        image.save(ROOT / f"mosaic-{month}.webp", quality=88, method=6)
        matches = [
            r
            for r in records
            if str(r.get("properties", {}).get("datetime", "")).startswith(month)
        ]
        reports[month] = {
            "date": start,
            "catalog_items": len(matches),
            "cloud_cover_values": [
                r.get("properties", {}).get("eo:cloud_cover") for r in matches
            ],
        }

    tile_w, tile_h = next(iter(images.values())).size
    sheet = Image.new("RGB", (tile_w * 3, tile_h + 40), "#17202a")
    draw = ImageDraw.Draw(sheet)
    for col, (month, image) in enumerate(images.items()):
        sheet.paste(image, (col * tile_w, 40))
        draw.text((col * tile_w + 12, 12), month, fill="white")
    sheet.save(ROOT / "contact-sheet.webp", quality=88)

    # Affine tilted-stack sketch: vertical depth axis is time, each plane retains
    # the exact RGB mosaic pixels. Perspective is illustrative, not a map view.
    canvas = Image.new("RGB", (1400, 900), "#111827")
    for i, (month, image) in enumerate(images.items()):
        layer = image.resize((760, 520))
        layer = layer.transform(
            (760, 520),
            Image.Transform.AFFINE,
            (1, 0.30, 0, 0.12, 1, 0),
            resample=Image.Resampling.BICUBIC,
        )
        x, y = 250 + i * 110, 170 + i * 155
        canvas.paste(layer, (x, y))
        ImageDraw.Draw(canvas).text((x + 12, y + 8), month, fill="white")
    canvas.save(ROOT / "tilted-seasonal-stack.webp", quality=88)
    (ROOT / "results.json").write_text(
        json.dumps(
            {
                "months": reports,
                "grid": {"crs": CRS, "bounds": BOUNDS, "resolution": RESOLUTION},
                "shape": [tile_h, tile_w],
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
