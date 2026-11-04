# Southwest Sentinel-2 assets

These flat review assets use the Earth Search `sentinel-2-c1-l2a` collection and the Southwest example's EPSG:3310 bounds, RGB bands, `eo:cloud_cover < 50` scene filter, ascending cloud-cover ordering, and first-valid source-order mosaic. The images are visual previews; they are not pixel-cloud-masked.

## Outputs

- `mosaic-2025-03.png`, `mosaic-2025-07.png`, `mosaic-2025-11.png`: three monthly mosaics.
- `datatake-2025-MM-{1,2}.png`: two selected multi-granule datatakes per month. They are a representative subset, not an exhaustive account of every monthly observation or each output pixel's provenance. White lines show the included item footprints.
- `contact-sheet.png`: all nine images in one flat review sheet.
- `array.html`: Xarray's actual `_repr_html_()` output for the monthly array at **300 m**, with the Xarray CSS and SVG symbols embedded for offline display. It is a lazy array repr, not a materialized raster.
- `footprints.svg`: CRS-colored footprints of precisely the selected datatake items; its legend counts are scoped to those displayed items.
- `manifest.json` / `results.json`: index query counts, selected item and datatake IDs, acquisition dates, grid and overview evidence, and runtime versions.
- `example.py`: short `lazycogs.open(...)` snippet used to define the monthly array.

## Grid and bounded rendering

The real array repr and overview-selection evidence are for the notebook's 300 m EPSG:3310 grid. **Every rendered PNG is a 1200 m display preview** over the same bounds and CRS. The preview grid is 938 x 924 pixels (transform `[1200, 0, -444000, 0, -1200, 500000]`). Monthly and datatake outputs are processed sequentially with `max_concurrent_reads=2`; no complete 300 m monthly raster is materialized to create these PNGs. However, the latest complete reproduction still peaked at 14,143,236 KiB RSS (`/usr/bin/time -v`) and took 24:55. This exceeds the requested 3 GB ceiling. The outputs are usable, but do not rerun the full script in a memory-constrained environment until this is resolved.

All images use the same fixed linear RGB display stretch (DN 0–5000 mapped to 0–255); alpha is transparent where all RGB bands are zero. The chosen months (March, July, November) are separated through the year. Their choices represent temporal spacing, not a claim of particular seasonal/cloud characteristics. Scene-level cloud filtering/ranking does not guarantee cloud-free pixels.

## Reproduce

From the repository root:

```sh
uv run python slides/foss4gna-2026/assets/southwest/reproduce.py
```

The script reuses `.cache/southwest-images/southwest-2025-full.parquet` when present and does not refresh it. The cache is the existing full-year 2025 index. Source COGs are read via anonymous public HTTPS using `obstore.HTTPStore` rooted at the same hostname in each original public object URL, preserving its object key. It does not discover or use AWS credentials. The only online work during a reproduction is catalogue/COG reading against public endpoints; no paid resources are required. The current end-to-end script has not met the requested peak-memory budget, as noted above.
