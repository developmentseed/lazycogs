# lazycogs: super lazy analysis of massive COG archives

## Format

25 minutes: about 20 minutes presenting, then 5 minutes discussing. Twelve slides, with one native click on slide 11 to replace the array definition with its May result. No custom animations, live queries, or live notebook execution. Keep the closing slide up during discussion.

Use the Development Seed theme retained in `../theme/`, following `~/workspace/devseed/ds-slidev-template/`. Its implementation uses Antarctica for headings/body, Shiki's GitHub light/dark code themes, and standard Slidev layouts. Local font assets and monospace fallbacks avoid runtime font downloads. Talk-local CSS only handles layout. Fenced Python blocks use Shiki rather than hand-styled `<pre>` elements.

## Story and pacing

| Slide | Time | Visible material |
|---|---|---|
| 1. Title | 0:00–0:20 | Combined lazycogs logo, subtitle, Henry Rodman / Development Seed, event panel |
| 2. I want to work with this as an array | 0:20–1:10 | Actual May Southwest notebook RGB plot |
| 3. GeoParquet holds metadata. COGs hold pixels. | 1:10–2:30 | STAC item fields, concrete public HLS partition URI, short rustac search, archive link and QR |
| 4. The useful array interface already exists | 2:30–3:45 | Short odc-stac load, spatial selection and compute; credit stackstac |
| 5. What if we also deferred finding the files? | 3:45–4:45 | Define → select → query assets → read pixels |
| 6. GTI worked. Packaging was the obstacle. | 4:45–6:00 | Actual prototype open pattern, red asset column, source link |
| 7. lazycogs: a year of HLS as an array | 6:00–8:30 | Literal grid/extent, 2025 daily RGB open, verified HTML repr |
| 8. Load one point | 8:30–10:30 | Literal EPSG:32615 point and real 2025 DN plot with gaps |
| 9. A new scene filter, then a small window | 10:30–12:45 | Explicit reopened filtered view, 5 km selection, matching seasonal RGB montage |
| 10. This time, prepare our own index | 12:45–14:15 | rustac.search_to from Earth Search to local Southwest GeoParquet |
| 11. Your scene selection. Your target grid. | 14:15–18:45 | Exact monthly Southwest open + notebook HTML; one click shows May selection/load and opening image |
| 12. Discussion | 18:45–20:00 | Logo, key message, installation, prominent docs link and local QR |

Speaker notes contain prerequisites, explanations, limitations, and transitions. Rehearse to 20 minutes; background runs to about six minutes. Do not add timing or scaling claims without new evidence.

## Central message

**The index does not decide which scenes you want. You do.**

The user chooses supported metadata filters, temporal grouping, extent, target CRS and resolution. There is no single imposed scene selection or precomputed mosaic. Changing filters, grouping, or resolution defines a new array. `.sel(...)` selects within an existing view.

Be fair to odc-stac and stackstac: their pixel reads are lazy, graph optimization can prune work, and they established the useful array interface. The motivation is up-front metadata enumeration and graph construction for a large view when the eventual request may be small. The conventional snippet illustrates that pattern; it is not a benchmark or a captured output.

Deferred asset discovery carries query criteria rather than the whole matching item collection. Queries happen for requested spatial chunks and time steps. Opening still inspects representative items, COG headers, and timestamps. This does not eliminate Dask graphs or index preparation.

## Source and code/output crosswalk

### STAC and HLS index access

Read `docs/notebooks/hls-s3.ipynb`, `hls-https.ipynb`, and `assets/hls/{README.md,manifest.json,reproduce.py}` before changing the examples.

Slide 3 shows the concrete public January object:

```text
https://s3.us-west-2.amazonaws.com/nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2025/month=1/HLSS30_2.0-2025-1.parquet
```

The month is unpadded. Exact objects are public; anonymous remote wildcard listing is not. The S3 notebook's wildcard path uses a client with bucket-listing privileges. The HTTPS notebook downloads exact objects to a local hive tree. The verified captures resolve the published Iceberg snapshot/manifests and cache twelve 2025 HLSS30 partitions. Someone still prepares and maintains that index. The archive link and QR point to https://maap-project.github.io/hls-stac-geoparquet-archive/.

### GTI

Sources actually present in `/home/henry/workspace/hrodmn/stac-gti-xarray/`: `demo.ipynb`, `Dockerfile`, `README.md`. Do not cite the absent `src/stac_gti_xarray/_core.py`.

The displayed `/tmp/tmp0umyk8_2/items_projected.parquet` is the projected filename under the saved notebook run's temporary directory. It is not a persistent file distributed with this deck. The snippet isolates one red-band open using `driver="GTI"` and `LOCATION_FIELD="assets.red.href"`; the Parquet may include other band asset columns too, and contains metadata, not raster pixels. The notebook's per-band symlink workaround stays in notes. `dt_filter` is the notebook loop's June 1 predicate with its inclusive next-midnight bound; its setup stays in notes. GTI selects/reprojects/mosaics; Python assembles dates and bands.

The prototype worked. Distributing the native build/runtime chain was the obstacle. The current Dockerfile starts from a GDAL Alpine image and builds rasterio from source; distinguish that from the speaker's earlier source-built GDAL experience.

### HLS array and results

`assets/hls/array.html` is the real verified 2025 baseline capture. Do not use the HTTPS notebook's stale 494-step saved output or the S3 notebook's older ascending-y output.

- HLSS30_2.0, EPSG:32615, bbox `(166020, 0, 834000, 9329010)`, 30 m, daily, B04/B03/B02, year 2025.
- Shape `(band,time,y,x)=(3,365,310967,22266)`, int16, nodata -9999. About 15 TB is logical dense-array size, not downloaded bytes or RAM.
- The capture used a local hive tree under `docs/notebooks/data/hls-stac-geoparquet-archive/v2/HLSS30_2.0/`, a hive-aware `DuckdbClient`, and a preconfigured Earthdata-authenticated HTTPS store. Slide code assumes the notebook working directory. The CQL2-text projection filter and date-only full-year interval are equivalent to the capture's CQL2-JSON and explicit full-day timestamp interval.
- `point-rgb.png` is the original plotted point output at `(576465,5196159)`: 123 valid RGB dates, 242 missing dates left as gaps, stored DNs. Do not substitute the later SVG marker additions, which do not reconstruct all raw observations. A point still reads compressed COG tiles.
- `nearby-rgb.png` is from a **separately reopened daily view** with scene cloud cover <30, not the baseline. Slide 9 shows this definition change. Its `hls_args` contains exactly the preceding open's index, bbox, CRS, resolution, daily grouping, bands, datetime, client and store, excluding the old filter; no new chunks or sortby override.
- The montage bounds are `(573965,5193659,578965,5198659)` in EPSG:32615. Dates are April 7, July 14, October 10. Display clips DN to 0–2000. Low scene cloud cover is not pixel cloud masking.

No live auth is needed to present. Reproduction requires NASA Earthdata credentials; direct HLS S3 pixels require compute in AWS us-west-2. Use HTTPS elsewhere. See the HLS asset scripts for the exact executed configurations.

### Southwest notebook

`docs/notebooks/demo_southwest_monthly.ipynb` is the primary source. `assets/southwest-notebook/` contains extracted saved outputs and cell/hash provenance. The old `assets/southwest/` March/July/November full-year, uint16, 1200 m previews are not used by this deck.

- Search: Earth Search `sentinel-2-c1-l2a`, March–June 2025, saved to `data/ca_spring_2025.parquet`. The literal geographic bbox is produced by `Transformer.from_crs(3310,4326,always_xy=True).transform_bounds(-444000,-609000,681000,500000)`. The notebook caches the result; `limit=100` is a page size, not the total item cap.
- Open: EPSG:3310; bbox `(-444000,-609000,681000,500000)`; resolution 300; monthly; red/green/blue; int16; scene cloud <50; ascending scene cloud order. Setup uses `lazycogs.store_for(..., skip_signature=True)`.
- `array.html` must be the actual **four-month** HTML output from cell 5, not the May selection from cell 6. Shape `(3,4,3697,3750)`, about 333 MB logical size.
- `mosaic-2025-05.png` is decoded directly from cell 11's saved PNG. Selection is `.chunk(x=1024,y=1024).sel(time="2025-05-01")`; loading uses `await ...load_async()`; plot uses RGB, DN 0–4000. The load cell has no saved execution count/output, so the notebook does not independently record that call's completion. The plot itself is a real saved output, not a fresh rerun.
- Monthly output is first-valid mosaicking in scene-cloud order, not a mean, median, or pixel mask. Multiple source projections are aligned to one target grid. COG overviews support the coarser grid; lazycogs currently samples nearest-neighbor.

## Presentation checks

Build with `cd slides && pnpm build`. Follow `../README.md` for local preview and publishing paths. Check all twelve slides plus slide 11's result state for readable code, clipping, the correct array repr, and bundled resources with external network access blocked.

Xarray reprs are imported verbatim into isolated iframes. `assets/deck/xarray.css` adapts the color variables and grid/list/attribute fixes from `docs/stylesheets/extra.css` without its MkDocs/Jupyter host selectors. The stylesheet is injected inside each frame, not just into the parent deck. Frame setup collapses Attributes using Xarray's native disclosure control so the coordinate summary fits; the original source markup remains unchanged. The two QR images are local SVGs, not runtime services.

Preserve the submitted proposal in `AGENTS.md` verbatim. Keep the scope caveats in closing notes: compatible STAC-indexed COGs, not a general raster engine. Closing docs URL is verified against `mkdocs.yml`: https://developmentseed.org/lazycogs/.
