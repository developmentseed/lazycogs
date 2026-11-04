# Notebook-derived Southwest May assets

These files are extracted from saved outputs in `docs/notebooks/demo_southwest_monthly.ipynb`; no notebook cells were rerun and no network data were fetched. The PNG is the saved May plot output, not a preview from the separate Southwest assets directories. `array.html` is the saved Xarray HTML representation of the May-selected DataArray, wrapped in a standalone document with the notebook's Xarray CSS and SVG symbols kept inline for offline use.

## Notebook configuration

The notebook configures the March–June 2025 query using local `data/ca_spring_2025.parquet`. Its projected AOI is EPSG:3310, `dst_bbox = (-444000, -609000, 681000, 500000)`, with `resolution=300`, `time_period="P1M"`, RGB bands, `dtype="int16"`, `filter="eo:cloud_cover < 50"`, and `sortby="eo:cloud_cover"`. The `rustac.search_to` call is conditional on the parquet not existing; it requests `sentinel-2-c1-l2a`, dates `2025-03-01/2025-06-30`, the transformed geographic bbox, and `limit=100`.

The exact May selection and plot source is:

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

The exact projected bbox passed to `rustac.search_to` after `Transformer.from_crs("epsg:3310", "epsg:4326", always_xy=True).transform_bounds(...)` is `(-125.38223191231621, 32.29621234930089, -111.75677519150827, 42.51937087572006)` (west, south, east, north).

## Extraction and caveats

Cell IDs, indices, execution counts, notebook/output hashes, and file hashes are in `provenance.json`. Image dimensions were checked by decoding the PNG; the result is 585 × 547 RGBA. `array.html` retains the actual output markup and self-contained Xarray stylesheet/SVG; it has no remote references and was checked offline for those resources.

The notebook's saved array representation reports four monthly coordinates (March–June), and the saved May selection reports date `2025-05-01`, RGB, `int16`, and a 300 m EPSG:3310 grid. The May plot is saved at cell 11 (execution count 8). The preceding `load_async()` cell has no execution count or saved output, so the notebook does not independently record that cell's successful execution; the saved plot is nevertheless the actual notebook output and is not represented as a fresh rerun. Scene-cloud filtering and ordering are as configured in the code; neither claim establishes pixel-level cloud masking.
