# lazycogs: super lazy analysis of massive COG archives

## Format

25 minutes: about 20 minutes presenting, then 5 minutes discussing. Sixteen slides, with one native click on slide 15 to replace the array definition with its May result. No live queries or live notebook execution. The logo-only introduction and closing slides periodically show googly eyes while active. Reduced-motion and print views keep the resting expression.

Use the Development Seed theme retained in `../theme/`, following `~/workspace/devseed/ds-slidev-template/`. Its implementation uses Antarctica for headings/body, Shiki's GitHub light/dark code themes, and standard Slidev layouts. Local font assets and monospace fallbacks avoid runtime font downloads. Talk-local CSS only handles layout. Fenced Python blocks use Shiki rather than hand-styled `<pre>` elements.

## Story and pacing

| Slide | Time | Visible material |
|---|---|---|
| 1. Title | 0:00–0:20 | Development Seed logo at top left, lazycogs wordmark and subtitle, Lena Delta image along the bottom, Henry Rodman / Development Seed event panel |
| 2. I want to work with this as an array | 0:20–1:10 | Actual May Southwest notebook RGB plot |
| 3. Search STAC, then hand the items to Xarray | 1:10–3:05 | Earth Search query via pystac-client → item collection → chunked odc-stac load and selection; credit stackstac |
| 4. What if we also defer STAC searches? | 3:05–4:05 | Orange brand background; define → select → query assets → read pixels |
| 5. Super duper | Transition at 4:05 | Image-only IT Crowd GIF from [Tenor](https://tenor.com/vZ9l.gif), bundled locally |
| 6. Cloud-native data needs cloud-native metadata | 4:05–4:25 | Cloud-native formats image from Pete Gadomski, STAC / stac-geoparquet introduction, source link |
| 7. STAC items as a queryable table | 4:25–4:45 | STAC JSON → GeoParquet diagram from Pete Gadomski, source link |
| 8. GTI worked. Packaging was the obstacle. | 4:45–6:00 | GDAL Raster Tile Index definition, prototype open, Parquet driver / source-build requirement |
| 9. lazycogs wakes up | 6:00–6:10 | Large animated logo on its own |
| 10. lazycogs: a year of HLS as an array | 6:10–8:30 | Literal grid/extent, 2025 daily RGB open, verified HTML repr |
| 11. Built on these tools | 8:30–9:00 | Three panels for obstore, async-geotiff, and rustac, with repo links over the Tanezrouft Basin background |
| 12. Load one point | 9:00–10:30 | Literal EPSG:32615 point and code for a date × RGB value table, with nodata masked |
| 13. A new scene filter, then a small window | 10:30–12:45 | Explicit reopened filtered view, 5 km selection, matching seasonal RGB montage |
| 14. This time, prepare our own index | 12:45–14:15 | rustac.search_to from Earth Search to local Southwest GeoParquet |
| 15. Your scene selection. Your target grid. | 14:15–18:45 | Exact monthly Southwest open + notebook HTML; one click shows May selection/load and opening image |
| 16. Discussion | 18:45–20:00 | Logo, bold “stay lazy”, prominent docs link and theme QR component; notes invite trials, feature requests and bug reports |

Speaker notes contain prerequisites, explanations, limitations, and transitions. Rehearse to 20 minutes; background runs to about six minutes. Do not add timing or scaling claims without new evidence.

## Central message

**The index does not decide which scenes you want. You do.**

The user chooses supported metadata filters, temporal grouping, extent, target CRS and resolution. There is no single imposed scene selection or precomputed mosaic. Changing filters, grouping, or resolution defines a new array. `.sel(...)` selects within an existing view.

Be fair to odc-stac and stackstac: their pixel reads are lazy, graph optimization can prune work, and they established the useful array interface. The motivation is up-front metadata enumeration and graph construction for a large view when the eventual request may be small. The conventional snippet illustrates that pattern; it is not a benchmark or a captured output.

Deferred asset discovery carries query criteria rather than the whole matching item collection. Queries happen for requested spatial chunks and time steps. Opening still inspects representative items, COG headers, and timestamps. This does not eliminate Dask graphs or index preparation.

## Source and code/output crosswalk

### STAC and HLS index access

Read `docs/notebooks/hls-s3.ipynb`, `hls-https.ipynb`, and `assets/hls/{README.md,manifest.json,reproduce.py}` before changing the examples.

Slides 5–6 introduce stac-geoparquet with unchanged images from Pete Gadomski's November 4, 2025 presentation and visible source links. Image URLs and attribution are recorded in `assets/stac-geoparquet/README.md`.

Slide 3 illustrates a STAC API → item collection → odc-stac workflow using pystac-client, Earth Search's `sentinel-2-c1-l2a` collection, January 2025, and common red/green/blue asset names. `item_collection()` retrieves matching metadata before the chunked `odc.stac.load()` call; pixel reads wait for compute. This is illustrative code, not a captured result or benchmark, and it is separate from the later HLS example.

The later HLS array uses a prepared GeoParquet index. Exact public partition objects are readable; anonymous remote wildcard listing is not. The S3 notebook's wildcard path uses bucket-listing privileges. The HTTPS notebook downloads exact objects to a local hive tree. The verified captures resolve the published Iceberg snapshot/manifests and cache twelve 2025 HLSS30 partitions. NASA MAAP maintains the archive: https://maap-project.github.io/hls-stac-geoparquet-archive/.

### GTI

Sources actually present in `/home/henry/workspace/hrodmn/stac-gti-xarray/`: `demo.ipynb`, `Dockerfile`, `README.md`. Do not cite the absent `src/stac_gti_xarray/_core.py`.

The slide adapts the notebook's temporary projected Parquet path to `items.parquet` and uses an illustrative May 2026 filter with an exclusive June 1 bound. No persistent index is distributed with this deck. The snippet isolates one red-band open using `driver="GTI"` and `LOCATION_FIELD="assets.red.href"`; the Parquet may include other band asset columns too, and contains metadata, not raster pixels. The notebook's per-band symlink workaround stays in notes. GTI selects/reprojects/mosaics; Python assembles dates and bands.

GTI means GDAL Raster Tile Index. Rasterio's PyPI wheels omit GDAL's Parquet vector driver, so this GeoParquet-backed workflow requires building rasterio from source against GDAL with Parquet support. The prototype worked. Distributing the native build/runtime chain was the obstacle. The current Dockerfile starts from a GDAL Alpine image and builds rasterio from source; distinguish that from the speaker's earlier source-built GDAL experience.

### HLS array and results

`assets/hls/array.html` is the real verified 2025 baseline capture. Do not use the HTTPS notebook's stale 494-step saved output or the S3 notebook's older ascending-y output.

- HLSS30_2.0, EPSG:32615, bbox `(166020, 0, 834000, 9329010)`, 30 m, daily, B04/B03/B02, year 2025.
- Shape `(band,time,y,x)=(3,365,310967,22266)`, int16, nodata -9999. About 15 TB is logical dense-array size, not downloaded bytes or RAM.
- The capture used a local hive tree under `docs/notebooks/data/hls-stac-geoparquet-archive/v2/HLSS30_2.0/`, a hive-aware `DuckdbClient`, and a preconfigured Earthdata-authenticated HTTPS store. Slide code assumes the notebook working directory. The CQL2-text projection filter and date-only full-year interval are equivalent to the capture's CQL2-JSON and explicit full-day timestamp interval.
- Slide 11 selects `(576465,5196159)` and shows code to mask -9999 to NaN and convert to a date-by-band pandas table. The verified capture had 123 valid RGB dates and 242 missing dates. Its raw array was not retained, so the slide shows four explicitly labeled pseudo-value rows to illustrate the table structure, not captured observations. The old `point-rgb.png` remains a provenance asset but is not displayed. A point still reads compressed COG tiles.
- `nearby-rgb.png` is from a **separately reopened daily view** with scene cloud cover <30, not the baseline. Slide 12 shows this definition change. Its `hls_args` contains exactly the preceding open's index, bbox, CRS, resolution, daily grouping, bands, datetime, client and store, excluding the old filter; no new chunks or sortby override.
- The montage bounds are `(573965,5193659,578965,5198659)` in EPSG:32615. Dates are April 7, July 14, October 10. Display clips DN to 0–2000. Low scene cloud cover is not pixel cloud masking.

No live auth is needed to present. Reproduction requires NASA Earthdata credentials; direct HLS S3 pixels require compute in AWS us-west-2. Use HTTPS elsewhere. See the HLS asset scripts for the exact executed configurations.

### Southwest notebook

`docs/notebooks/demo_southwest_monthly.ipynb` is the primary source. `assets/southwest-notebook/` contains extracted saved outputs and cell/hash provenance. The old `assets/southwest/` March/July/November full-year, uint16, 1200 m previews are not used by this deck.

- Search: Earth Search `sentinel-2-c1-l2a`, March–June 2025, saved to `data/ca_spring_2025.parquet`. The literal geographic bbox is produced by `Transformer.from_crs(3310,4326,always_xy=True).transform_bounds(-444000,-609000,681000,500000)`. The notebook caches the result; `limit=100` is a page size, not the total item cap.
- Open: EPSG:3310; bbox `(-444000,-609000,681000,500000)`; resolution 300; monthly; red/green/blue; int16; scene cloud <50; ascending scene cloud order. Setup uses `lazycogs.store_for(..., skip_signature=True)`.
- `array.html` must be the actual **four-month** HTML output from cell 5, not the May selection from cell 6. Shape `(3,4,3697,3750)`, about 333 MB logical size.
- `mosaic-2025-05.png` is decoded directly from cell 11's saved PNG. Selection is `.chunk(x=1024,y=1024).sel(time="2025-05-01")`; loading uses `await ...load_async()`; plot uses RGB, DN 0–4000. The load cell has no saved execution count/output, so the notebook does not independently record that call's completion. The plot itself is a real saved output, not a fresh rerun.
- The slide specifies `mosaic_method=lazycogs.FirstMethod`, making the notebook's default explicit. Monthly output is first-valid mosaicking in scene-cloud order, not a mean, median, or pixel mask. Multiple source projections are aligned to one target grid. COG overviews support the coarser grid; lazycogs currently samples nearest-neighbor.

## Presentation checks

Build with `cd slides && pnpm build`. Follow `../README.md` for local preview and publishing paths. Check all fifteen slides plus slide 14's result state for readable code, clipping, the correct array repr, and bundled resources with external network access blocked.

Xarray reprs are imported verbatim into isolated iframes. `assets/deck/xarray.css` adapts the color variables and grid/list/attribute fixes from `docs/stylesheets/extra.css` without its MkDocs/Jupyter host selectors. The stylesheet is injected inside each frame, not just into the parent deck. Frame setup collapses Attributes using Xarray's native disclosure control so the coordinate summary fits; the original source markup remains unchanged. The closing QR code uses the theme's `CurrentUrlQRCode` component with explicit URLs and `slidev-addon-qrcode`. The addon renders it locally using a bundled DS symbol, without a remote QR service.

Preserve the submitted proposal in `AGENTS.md` verbatim. Keep the scope caveats in closing notes: compatible STAC-indexed COGs, not a general raster engine. Closing docs URL is verified against `mkdocs.yml`: https://developmentseed.org/lazycogs/.
