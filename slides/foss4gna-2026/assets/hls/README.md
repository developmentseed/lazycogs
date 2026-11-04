# HLS slide assets

Current, verified outputs are ready for layout:

- `array.html` — actual HTML `_repr_html_()` from the current 2025 baseline open, with Xarray's stylesheet embedded for offline display.
- `coverage.svg` and `selection.svg` — 2025 source MGRS tile footprints, overlaid with the selected point and a 5 km square window. Footprints are time-unioned across the year; this is spatial item coverage, not cloud-free pixel coverage.
- `plan.txt` and `plan.svg` — actual `subset.lazycogs.explain()` result for the same 5 km window and seasonal montage dates.
- `point-rgb.svg` / `point-rgb.png` — current 2025 point RGB daily time series. Nodata dates are retained as gaps.
- `nearby-rgb.png` — actual 5 km RGB windows at three low-scene-cloud dates, selected near spring, summer, and fall.
- `nearby-dates.json` — item IDs, scene cloud cover, selected dates, clip range, and bounded plan count for the montage.
- `snippets.md` — executed baseline and alternate monthly scene-cloud-filter open examples.
- `reproduce.py`, `capture_assets.py`, `capture_nearby_rgb.py` — bounded reproduction scripts. `manifest.json` records source, commands, checks, output scope, and limitations.

## Source and current-index reconciliation

NASA MAAP's current public GeoParquet archive is Iceberg-backed. The public pointer is [`latest.metadata.json`](https://s3.us-west-2.amazonaws.com/nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/iceberg/metadata/latest.metadata.json), snapshot `69769654033930513`. Its manifest lists monthly parquet files with **unpadded** months. For example, the current January object is `.../year=2025/month=1/HLSS30_2.0-2025-1.parquet`; the old notebook's padded `month=01/HLSS30_2.0-2025-01.parquet` gets 403. Bucket listing remains denied, but the documented pointer, Avro manifests, and exact public data objects are readable by HTTPS. `reproduce.py` follows the snapshot manifests and fetches exactly the 12 2025 `HLSS30_2.0` partitions, not the other collection or year. The files total 665,392,268 compressed bytes / 2,715,469 records.

The array is `HLSS30_2.0`, `EPSG:32615`, 30 m, daily, B04/B03/B02, with explicit 2025 start/end. The actual repr reports `(band,time,y,x)=(3,365,310967,22266)`, `int16`, 7,581,770,388,090 logical values and 15,163,540,776,180 logical bytes. That's dense-array size, not downloaded or resident bytes. A current sample item's B04 asset reports wavelength/common-name metadata but no `raster:bands` scale or unit. The arrays/plots therefore show stored integer digital numbers (DN), **not reflectance-scaled values**. The RGB montage clips DN to 0–2000; no physical reflectance scale is asserted.

The current sample COG header agrees with the baseline output dtype `int16` and nodata `-9999`; the 2025 STAC feature has `proj:epsg=32615` and `eo:cloud_cover`. Both daily baseline and monthly `eo:cloud_cover < 50` opens ran successfully against the current index. The monthly view defines a separate array and sorts by scene cloud cover; it does not cloud-mask pixels.

## Point and seasonal image story

At point `(576465, 5196159)`, 123 of 365 daily slots have all three RGB bands non-nodata; the other 242 remain gaps in the plot. The nearby image strip uses a 5 km square centered on the same point. To make the seasonal comparison interpretable, candidates are restricted to scene cloud cover below 30%, then selected nearest April 15, July 15, and October 10. The actual dates are 2025-04-07 (4% scene cloud), 2025-07-14 (6%), and 2025-10-10 (0%). In this neighborhood, the RGBs show mostly brown/low-green early-season fields, green summer vegetation, and browner autumn fields. This is an illustrative scene sequence, not a quantitative crop-phenology result; low scene cloud cover does not guarantee clear pixels. The plan for this 5 km / three-date subset has 9 chunk reads and 15 planned COG reads; no full array was computed.

Current STAC item/asset metadata does not declare a physical DN scale or unit. We label plotted values as stored DN and avoid claiming surface reflectance values. See `manifest.json` for metadata evidence and source commands.

## HTTPS and S3 locations

The existing local Earthdata authentication flow succeeded for bounded HTTPS HLS COG reads; secrets and authorization headers were not logged or persisted. Direct HLS S3 pixel access requires compute in AWS `us-west-2`; use HTTPS elsewhere. Metadata reproduction uses unsigned public HTTPS objects and does not use AWS credential discovery.
