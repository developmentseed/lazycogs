# Southwest seasonal imagery spike

**Status: blocked before image export.** The public Earth Search query succeeded and returned 9,578 Sentinel-2 items for the notebook's March–June 2025 AOI, but opening the public S3 COGs failed during credential discovery: obstore attempted the EC2 metadata endpoint (`169.254.169.254`) and timed out. No imagery or preview is included; this directory does not fabricate substitutes. See `manifest.json` for the verified catalog facts and exact failure.

## Reference workflow

The source is `docs/notebooks/demo_southwest_monthly.ipynb`: Earth Search collection `sentinel-2-c1-l2a`, dates 2025-03-01 through 2025-06-30, EPSG:3310 bounds `(-444000, -609000, 681000, 500000)`, 300 m target grid, RGB assets `red/green/blue`, `eo:cloud_cover < 50`, sorted ascending by cloud cover, and monthly first-valid mosaicking. The scene cloud filter/rank is not pixel-level cloud masking. Index is local scratch at `.cache/southwest-images/southwest-2025.parquet` and is not committed.

The attempted preview workflow is `reproduce.py`. Its 1200 m grid was chosen to limit this review-only pixel read to roughly one-sixteenth the notebook grid. Once public COG access is available, it is intended to render March, May, and June RGB mosaics with one fixed display stretch. These three dates span early spring to early summer; actual visual coverage/seasonality selection remains pending pixel inspection. It also reports item-level `s2:datatake_id` for pass grouping, but orbit ingredient renders are not implemented and must not be inferred from tiles.

## Reproduction

```sh
uv run python slides/foss4gna-2026/spikes/southwest-images/reproduce.py
```

The catalog metadata query completed in this environment. The subsequent imagery read did not. Do not use this spike as final slide material until the COG reads, month selection, orbit/pass views, and contact/tilted previews are completed and inspected. The presentation slides were not changed.
