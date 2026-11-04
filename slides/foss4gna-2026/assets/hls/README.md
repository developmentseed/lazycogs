# HLS slide assets: BLOCKED

No slide-ready HLS assets are supplied in this directory. The requested array repr, coverage and selection maps, explain plan, RGB time series, nearby RGB images, and verified alternate snippet all require a reproducible query/read against the HLS source. The HTTPS notebook states that the COG assets require an Earthdata Login bearer token. No task-specific token or already-configured authenticated HTTPStore was provided. I did not inspect credential stores, discover credentials, or fetch HLS pixels. Do not use this directory as if the requested slide assets were available.

The existing `docs/notebooks/hls-https.ipynb` output for its array represents 2025 through 2026-05-09 (494 daily coordinates), so it does not satisfy the requested 2025-only scope. The saved S3 notebook output does show 2025, but the notebook explicitly sets up credentials and accesses `lp-prod-protected`; that is not task-authorized provenance for fresh pixel work. Neither output is substituted here for verified reproduction.

The requested public-metadata/HTTPS path also requires authenticated COG reads: its `HTTPStore` is explicitly configured with `Authorization: Bearer <token>` in the notebook. Publicly reachable GeoParquet metadata does not make those pixel assets public.

When access is explicitly provisioned, reproduce under this interface and replace this blocked status with actual outputs and provenance:

- `array.html` — actual offline Xarray `_repr_html_()` with original Xarray CSS/assets embedded
- `coverage.svg`, `selection.svg` — actual coverage and point/window overlays
- `plan.txt`, `plan.svg` or `plan.png` — actual `subset.lazycogs.explain()` result
- `point-rgb.svg` or `point-rgb.png`, `nearby-rgb.png` — actual pixel results with gaps preserved
- `snippets.md` — only executed and verified baseline/alternate opens
- `manifest.json` — source, exact commands, checks, dimensions, scope, units, and provenance

The S3 path has a location restriction: direct HLS S3 access requires compute in AWS `us-west-2`; use HTTPS elsewhere. Keep auth, grid discovery, and storage setup behind named setup variables in any slide code.
