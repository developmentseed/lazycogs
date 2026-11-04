# ruff: noqa: INP001
"""Export provenance-tracked HLS outputs already saved in the source notebook."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import xarray

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "docs/notebooks/hls-s3.ipynb"
ASSET_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Export saved notebook outputs and a clearly scoped grid-selection map."""
    notebook = json.loads(NOTEBOOK.read_text())
    cells = notebook["cells"]
    array_cell = cells[15]
    image_cell = cells[21]
    array_html = "".join(array_cell["outputs"][0]["data"]["text/html"])
    css_path = Path(xarray.__file__).parent / "static/css/style.css"
    stylesheet = css_path.read_text()
    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<style>{stylesheet}</style></head><body>{array_html}</body></html>"
    )
    (ASSET_DIR / "array.html").write_text(document)
    png = base64.b64decode(image_cell["outputs"][1]["data"]["image/png"])
    (ASSET_DIR / "nearby-rgb.png").write_bytes(png)

    # Coordinates and bounds are from the executed HLS S3 notebook. The map
    # shows the requested grid extent, not observed source-scene coverage.
    west, south, east, north = 166020, 0, 834000, 9329010
    point_x, point_y = 576465, 5196159
    win_west, win_south, win_east, win_north = 550000, 5150000, 700000, 5250000
    left, top, width, height = 58, 54, 334, 232
    scale_x = width / (east - west)
    scale_y = height / (north - south)
    point = (left + (point_x - west) * scale_x, top + (north - point_y) * scale_y)
    window = (
        left + (win_west - west) * scale_x,
        top + (north - win_north) * scale_y,
        (win_east - win_west) * scale_x,
        (win_north - win_south) * scale_y,
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="450"
  height="350" viewBox="0 0 450 350">
<rect width="450" height="350" fill="#fff"/>
<text x="225" y="24" text-anchor="middle" font-family="sans-serif"
  font-size="15">HLS requested grid extent - EPSG:32615</text>
<rect x="{left}" y="{top}" width="{width}" height="{height}"
  fill="#e7eef1" stroke="#334155" stroke-width="1.5"/>
<rect x="{window[0]:.1f}" y="{window[1]:.1f}" width="{window[2]:.1f}"
  height="{window[3]:.1f}" fill="#f4b942" fill-opacity=".42"
  stroke="#a45c00" stroke-width="2"/>
<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="5"
  fill="#c62828" stroke="white" stroke-width="1.5"/>
<text x="{left}" y="304" font-family="sans-serif" font-size="12">
  Target bounds: 166,020-834,000 E; 0-9,329,010 N</text>
<text x="{left}" y="323" font-family="sans-serif" font-size="11"
  fill="#704800">Window outline; red point; not observed scene coverage</text>
</svg>"""
    (ASSET_DIR / "coverage.svg").write_text(svg)
    (ASSET_DIR / "selection.svg").write_text(svg)
    LOGGER.info("Exported saved HLS outputs and requested-grid selection map")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
