# Notebook-derived Southwest assets

These files are extracted from saved outputs in `docs/notebooks/demo_southwest_monthly.ipynb`; no notebook cells were rerun and no network data were fetched. The PNG is the saved May plot output, not a preview from the separate Southwest assets directories. `array.html` contains the actual saved Xarray HTML output beside the monthly `lazycogs.open(...)` call (cell 5): the four-month March–June array, not the May-selected repr. Its original output markup, CSS, and SVG are retained inside a minimal standalone HTML document for offline display.

## Notebook configuration

The notebook uses local `data/ca_spring_2025.parquet`. Its projected AOI is EPSG:3310, `dst_bbox = (-444000, -609000, 681000, 500000)`, with `resolution=300`, `time_period="P1M"`, RGB bands, `dtype="int16"`, `filter="eo:cloud_cover < 50"`, and `sortby="eo:cloud_cover"`. The `rustac.search_to` call is conditional on the parquet not existing; it requests `sentinel-2-c1-l2a`, dates `2025-03-01/2025-06-30`, `bbox=bbox_4326`, and `limit=100`. `bbox_4326` is computed from the projected bbox with `Transformer.from_crs(dst_crs, "epsg:4326", always_xy=True).transform_bounds(*dst_bbox)`.

The exact geographic bbox passed as `bbox=bbox_4326` to `rustac.search_to` is `(-125.38223191231621, 32.29621234930089, -111.75677519150827, 42.51937087572006)` (west, south, east, north). This is distinct from the projected `dst_bbox` above.

The notebook's May selection and plot source is:

```python
ca_may = ca_monthly.chunk(x=1024, y=1024).sel(time="2025-05-01")
ca_may

_ = await ca_may.load_async()

ca_may.plot.imshow(
    rgb="band",
    vmin=0,
    vmax=4000,
    size=6,
    aspect=ca_may.shape[2] / ca_may.shape[1],
)
```

## Extraction and caveats

Cell IDs, indices, execution counts, source/output hashes, and file hashes are in `provenance.json`. `array.html` is the saved Xarray HTML output from cell 5 (four monthly coordinates, March–June; 333 MB, `int16`; shape `(band: 3, time: 4, y: 3697, x: 3750)`; EPSG:3310 at 300 m). The HTML payload and its inline Xarray CSS/SVG are copied unchanged from that output; only an outer standalone-document wrapper is added. The May selection has its own repr at cell 6, but it is not what `array.html` contains.

The PNG is the saved May plot at cell 11 (execution count 8), 585 × 547 RGBA. The preceding `load_async()` cell has no execution count or saved output, so the notebook does not independently record that cell's successful execution; the saved plot is the actual notebook output and was not regenerated here. Scene-cloud filtering and sorting are as configured in code; neither proves pixel-level cloud masking.
