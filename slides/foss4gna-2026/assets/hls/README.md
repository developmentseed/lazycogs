# HLS slide assets: partial; remaining assets BLOCKED

This directory contains only provenance-supported outputs and clearly labeled limits. Layout interface is available now:

- `array.html`: saved Xarray HTML repr with Xarray stylesheet embedded for offline viewing.
- `coverage.svg` and `selection.svg`: same map of the requested UTM grid bounds and the saved notebook point/window. It explicitly says this is not observed scene coverage.
- `nearby-rgb.png`: actual saved RGB output from the HLS notebook, showing ten September 2025 subset dates. It is not a synthetic image.
- `snippets.md`: executed baseline configuration plus a candidate alternate marked unverified/BLOCKED.
- `extract_saved_outputs.py`: extracts the saved HTML/image outputs and makes the target-grid selection map. It does not query data or read pixels.
- `manifest.json`: provenance and blocked-item status.

## Source and verification limits

The baseline array HTML came from the real saved result in `docs/notebooks/hls-s3.ipynb`, cell 15. Its executed code selects collection `HLSS30_2.0`, explicit dates 2025-01-01 through 2025-12-31, `EPSG:32615`, 30 m, daily steps, and B04/B03/B02. The saved repr reports shape `(3, 365, 310967, 22266)`, `int16`, 7,581,770,388,090 values (15,163,540,776,180 logical bytes). This is historical notebook output, not a fresh rerun.

The saved September RGB figure is from that notebook's `subset.plot.imshow(...)` output (cell 21), where a 15 km by 10 km window was plotted for ten non-nodata days in September 2025. The notebook uses `vmin=0`, `vmax=2000`; the RGB values are raw `int16` HLS band digital numbers, not physical reflectance. The saved S3 notebook's point selection only covers September 2025, not the full year, so the requested year-long point time series is unavailable.

Existing local Earthdata auth succeeded using the HTTPS notebook's existing token path, without logging or exposing the token. A bounded authenticated HEAD to a documented HTTPS HLS COG succeeded (HTTP 200). However the unsigned public MAAP parquet object needed to query current spatial metadata returned HTTP 403 (`PermissionDeniedError`). No protected S3/AWS credential discovery was used. Thus no current index query, current `eo:cloud_cover` support check, alternate-open run, or `subset.lazycogs.explain()` was possible.

`coverage.svg`/`selection.svg` show the requested array grid rectangle and saved point/window only. They do not claim source-scene coverage. Plan assets and point-series assets remain BLOCKED rather than replaced with fake output.

Direct HLS S3 access requires compute in AWS `us-west-2`; use HTTPS elsewhere. Keep auth, grid inspection, and storage configuration behind named variables in slide code.
