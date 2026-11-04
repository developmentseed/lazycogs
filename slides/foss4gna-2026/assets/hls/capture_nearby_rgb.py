"""Capture seasonally separated RGB windows from verified low-cloud HLS dates."""

# ruff: noqa: INP001

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
from capture_assets import (
    ASSET_DIR,
    BANDS,
    DATETIME,
    INDEX,
    POINT_XY,
    WINDOW_HALF_SIZE_M,
    obtain_store,
)
from pyproj import Transformer
from rustac import DuckdbClient

import lazycogs

LOGGER = logging.getLogger(__name__)
SEASONS = (date(2025, 4, 15), date(2025, 7, 15), date(2025, 10, 10))
MAX_PLAN_COG_READS = 250


def main() -> None:
    """Select low-cloud dates near three seasons and read only a 5 km window."""
    longitude, latitude = Transformer.from_crs(
        "EPSG:32615",
        "EPSG:4326",
        always_xy=True,
    ).transform(*POINT_XY)
    bbox = (longitude - 0.3, latitude - 0.3, longitude + 0.3, latitude + 0.3)
    predicate = {
        "op": "and",
        "args": [
            {"op": "=", "args": [{"property": "proj:epsg"}, "32615"]},
            {"op": "<", "args": [{"property": "eo:cloud_cover"}, 30]},
        ],
    }
    candidates = DuckdbClient(use_hive_partitioning=True).search(
        str(INDEX),
        bbox=bbox,
        datetime=DATETIME,
        filter=predicate,
        exclude=["assets", "links", "stac_extensions", "bbox", "collection"],
    )
    by_date: dict[date, dict] = {}
    for item in candidates:
        acquired = datetime.fromisoformat(item["properties"]["datetime"]).date()
        previous = by_date.get(acquired)
        cloud_cover = item["properties"]["eo:cloud_cover"]
        if previous is None or cloud_cover < previous["properties"]["eo:cloud_cover"]:
            by_date[acquired] = item

    selected = []
    used: set[date] = set()
    for season in SEASONS:
        acquired = min(
            (candidate for candidate in by_date if candidate not in used),
            key=lambda candidate: (
                abs((candidate - season).days),
                by_date[candidate]["properties"]["eo:cloud_cover"],
            ),
        )
        used.add(acquired)
        selected.append(by_date[acquired])

    seasonal_filter = {
        "op": "and",
        "args": [
            {"op": "=", "args": [{"property": "proj:epsg"}, "32615"]},
            {"op": "<", "args": [{"property": "eo:cloud_cover"}, 30]},
        ],
    }
    common = {
        "href": str(INDEX),
        "bbox": (166020, 0, 834000, 9329010),
        "crs": "EPSG:32615",
        "resolution": 30,
        "bands": BANDS,
        "filter": seasonal_filter,
        "datetime": DATETIME,
        "duckdb_client": DuckdbClient(use_hive_partitioning=True),
        "store": obtain_store(),
    }
    imagery = lazycogs.open(**common, time_period="P1D")
    selected_dates = np.array(
        [item["properties"]["datetime"][:10] for item in selected],
        dtype="datetime64[s]",
    )
    x, y = POINT_XY
    selected_window = imagery.sel(
        x=slice(x - WINDOW_HALF_SIZE_M, x + WINDOW_HALF_SIZE_M),
        y=slice(y + WINDOW_HALF_SIZE_M, y - WINDOW_HALF_SIZE_M),
        time=selected_dates,
    )
    plan = selected_window.lazycogs.explain()
    if plan.total_cog_reads > MAX_PLAN_COG_READS:
        raise RuntimeError(
            f"Seasonal RGB plan exceeds read cap: {plan.total_cog_reads}",
        )
    (ASSET_DIR / "plan.txt").write_text(plan.summary() + "\n")
    lines = [
        wrapped_line
        for line in plan.summary().splitlines()
        if line and not line.startswith("Parquet:")
        for wrapped_line in wrap(line, width=58)
    ][:15]
    rows = "".join(
        f'<text x="18" y="{40 + index * 19}" font-family="monospace" '
        f'font-size="11">{line}</text>'
        for index, line in enumerate(lines)
    )
    (ASSET_DIR / "plan.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="450" height="340" '
        'viewBox="0 0 450 340"><rect width="450" height="340" fill="white"/>'
        '<text x="18" y="22" font-family="sans-serif" font-size="15">'
        "subset.lazycogs.explain() - actual summary</text>" + rows + "</svg>",
    )
    selected_values = (
        selected_window.compute().transpose("time", "y", "x", "band").values
    )
    nodata = imagery.attrs.get("_FillValue", -9999)
    scaled = np.clip(selected_values.astype(np.float32) / 2000, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), layout="constrained")
    for index, axis in enumerate(axes):
        image = scaled[index]
        image[np.any(selected_values[index] == nodata, axis=-1)] = 0
        axis.imshow(image, interpolation="nearest")
        axis.set_title(str(selected_dates[index])[:10], fontsize=9)
        axis.axis("off")
    fig.savefig(ASSET_DIR / "nearby-rgb.png", dpi=120)
    plt.close(fig)

    date_record = [
        {
            "datetime": item["properties"]["datetime"],
            "eo:cloud_cover_percent": item["properties"]["eo:cloud_cover"],
            "item_id": item["id"],
        }
        for item in selected
    ]
    nearby_dates = {
        "selection": (
            "nearest low-scene-cloud-cover date to 2025-04-15, 2025-07-15, 2025-10-10"
        ),
        "candidate_filter": "eo:cloud_cover < 30%",
        "candidates": len(candidates),
        "dates": date_record,
        "window_m": WINDOW_HALF_SIZE_M * 2,
        "plan_chunk_reads": plan.total_chunk_reads,
        "plan_cog_reads": plan.total_cog_reads,
        "rgb_clip_dn": [0, 2000],
        "nodata": nodata,
    }
    (ASSET_DIR / "nearby-dates.json").write_text(
        json.dumps(nearby_dates, indent=2) + "\n",
    )
    manifest_path = ASSET_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seasonal_rgb"] = nearby_dates
    manifest["seasonal_rgb"]["scene_cloud_filter"] = "eo:cloud_cover < 30%"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Captured three low-cloud seasonal windows in a 5 km extent")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
