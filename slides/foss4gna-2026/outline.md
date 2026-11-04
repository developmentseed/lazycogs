# lazycogs: super lazy analysis of massive COG archives

## Format

- 25-minute slot: 20-minute presentation, then 5 minutes for discussion.
- 12 main slides. Combine the desired-interface and deferred-discovery concepts into one slide; keep plan inspection dedicated. Slides 1–4 cover background and inspiration in 5 minutes. Timings are provisional; rehearse the opening animation and retain pacing slack.
- Minimal visible prose: a short heading, code, an image, or a diagram. No narrative bullet lists on screen.
- Put context, caveats, and transitions in Slidev speaker notes. The notes below are not slide copy.
- Use prepared outputs and figures; do not depend on live network access during the talk.
- Center the examples on year-long HLS over an entire UTM zone and Southwest Sentinel-2 across multiple source projections. Pair each `lazycogs.open(...)` call with the actual Xarray HTML DataArray repr, not a plain-text repr or a hand-built summary. Omit auth, grid inspection, and other setup from visible code; link the full notebooks.

## Story

“Query satellite imagery collections like everyone wants to: as a big array.” Open the content with a grand Southwest mosaic, reveal seasonally separated monthly views and their overlapping orbit-pass ingredients, then reassemble the mosaics. Establish the desired interface: choose filters, temporal grouping, and a target grid, then select a small window without preparing a new raster product. The main design goal is to carry query criteria rather than the entire matching item collection: defer asset discovery as well as pixel reads until computation needs them. Keep this opening implementation-neutral; introduce the working GTI experiment and its packaging obstacle before introducing lazycogs by name with the HLS coverage map. Show a year's HLS granules for an entire UTM zone as one lazy array, then follow a small selection through asset discovery and pixel reads. Return to the Southwest imagery to demonstrate alignment across source projections. Close with the scope tradeoffs and discussion.

## Main deck

### 1. Title and vision — 0:00–0:30

**On screen:** the combined relaxed-squares and “lazycogs” SVG (`../../logo-text.svg`) in place of a separate title. No imagery on the title slide. Use the event/presenter panel layout and typography from the collection-discovery deck: Henry Rodman only; FOSS4GNA 2026; Sacramento, CA; 2026-11-03. Subtitle matches the remainder of the submitted session title:

> super lazy analysis of massive COG archives

**Speaker notes:** Development Seed; introduce the vision: “Query satellite imagery collections like everyone wants to: as a big array.” The array is a view over existing COGs, not a requirement to rewrite the archive.

### 2. From overlapping observations to one mosaic — 0:30–1:30

**On screen:** staged visual sequence using the Southwest Sentinel-2 example:

1. A grand monthly RGB mosaic, viewed straight on, with almost no text.
2. Rotate around the vertical axis and separate three seasonally separated monthly mosaics into parallel planes to reveal time; keep month labels visible.
3. Replace the monthly mosaics with a small, explicitly representative selection of their real source orbit-pass imagery, exposing gaps and partial overlaps. Show acquisition dates and thin footprint outlines; label the pass selection as illustrative rather than an exhaustive pixel-by-pixel decomposition.
4. Reassemble the monthly mosaics, leaving the seasonal time stack visible.

**Assets/presentation:** generate flat mosaics and flat orbit-pass layers on a consistent extent/grid with consistent RGB scaling and transparent nodata where appropriate. Do rotation, stacking, transitions, and click-driven reveals in Slidev using CSS 3D transforms, not baked-in Python perspective images. Keep labels separable where practical. Choose a readable static fallback for PDF export. Select three seasonally separated months after inspecting real coverage and cloudiness; do not treat provisional month choices as verified. Use public HTTPS for the image-generation spike. No code on this slide.

**Speaker notes:** The archive does not contain this ready-made mosaic: it contains observations acquired at different times with overlapping footprints. Each monthly mosaic summarizes observations within that month; its source swaths are not simultaneous and do not contribute equally. The animation explains sources becoming a mosaic, not the provenance of every output pixel. Scene-level cloud ranking does not guarantee cloud-free output. A Sentinel-2 granule is a tile, not an entire orbit swath: group real granules by verified acquisition/pass identity and disclose any representative subset. Return to these same Southwest images in the later worked example.

**Source:** `../../docs/notebooks/demo_southwest_monthly.ipynb`; image-generation spike assets and provenance to be reviewed before integration.

### 3. Define the view. Discover when needed. — 1:30–3:15

**On screen:** an existing STAC GeoParquet index feeds two conceptual array views: daily with all observations, and monthly with a scene-cloud-cover filter. Use labels rather than placeholder API calls. Label monthly output as a first-valid mosaic ordered by scene cloud cover, not an unspecified aggregation. Carry forward the seasonal stack as an array with time/y/x axes and RGB bands; highlight a small spatial window through it. A familiar Xarray selection anchors the desired interface:

```python
subset = imagery.sel(
    time=slice("2025-03", "2025-06"),
    x=slice(xmin, xmax),
    y=slice(ymax, ymin),
)
```

The descending y slice assumes a north-up grid. No `lazycogs.open(...)` or implementation/library names yet; this is the desired interface. Focus the visual on selecting a small window, not distributing whole-array analysis.

**Speaker notes:** Different applications need different filters, daily/monthly/seasonal grouping, target grids, and extents. Make array views easy and fast to define without assuming a single precomputed product or fetching pixels prematurely. Fast opening is a design goal to demonstrate, not an unconditional guarantee. Scene cloud-cover filtering is not a pixel cloud mask. Credit stackstac and odc-stac for establishing the array pattern. Briefly describe the original large-scale project and its task-graph and API-query obstacles as project experience, not universal library limitations; keep chunkwise distributed processing in narration.

**Source:** speaker's experience; `../../docs/notebooks/lazycogs-odc-stac.ipynb` for a concrete comparison workflow.

**Second reveal on this same slide:** filters + temporal grouping + target grid → lazy array view → compute a selected window → discover matching assets from the existing STAC GeoParquet index → read pixels. Keep archive footprints muted until the requested window reveals its matching assets. No implementation/library names; do not label rustac here.

**Visible distinction:** index preparation → array opening → computation. Someone must prepare or publish searchable metadata first. HLS uses an existing index; the Southwest workflow prepares one from a STAC search. The benefit is avoiding enumeration and retention of the entire matching STAC item collection for each array view, not eliminating index preparation.

**Additional speaker notes:** carry query criteria and defer asset discovery alongside pixel reads until computation needs them. Opening still needs representative-item, COG-header, and timestamp metadata inspection. Changing filters, grouping, or target grid defines a new array via another open; `.sel(...)` selects within an existing view. Seasonally separated monthly examples do not establish arbitrary calendar-season grouping support. Do not imply that lazycogs eliminates Dask graphs or guarantees arbitrary distributed-worker deployment.

### 4. GTI worked. Packaging was the obstacle. — 3:15–5:00

**On screen:** a short, verified `rioxarray.open_rasterio` example opening `GTI:/path/to/stac.parquet`, highlighting `FILTER` and `LOCATION_FIELD`. Beside it, a compact build chain: GDAL with GTI + Parquet → rasterio built against it → rioxarray. Include a visible link beneath the code: [github.com/hrodmn/stac-gti-xarray](https://github.com/hrodmn/stac-gti-xarray).

**Speaker notes:** The prototype worked: GTI supplied spatial source selection, reprojection, and mosaicking; the Python layer assembled dates and bands. SQL filtering through an open option was useful. You built GDAL from source with Parquet/GeoParquet support, then built rasterio against it. Distributing that native build chain and runtime linkage was the obstacle. The current prototype Dockerfile starts from a GDAL image and compiles rasterio against it. This motivated a tightly scoped, GDAL-free implementation, not a claim that GTI could not do the raster work.

**Sources:** [hrodmn/stac-gti-xarray](https://github.com/hrodmn/stac-gti-xarray): `README.md`, `Dockerfile`, `docs/poc.ipynb`, `src/stac_gti_xarray/_core.py`. Local checkout: `/home/henry/workspace/hrodmn/stac-gti-xarray`.

### 5. A year of HLS, an entire UTM zone — 5:00–5:45

**On screen:** introduce lazycogs by name with a map-first view of UTM zone 15N and HLS coverage, plus a compact caption: “2025 · 30 m · RGB”. Show the existing NASA MAAP HLS STAC GeoParquet archive as the metadata source. Save the open call and array repr for the following slide.

**Speaker notes:** Introduce the scale before showing code: define an array covering the northern UTM-zone extent and a year of HLS observations, then request only what the analysis needs. The reference example selects `HLSS30_2.0` (Sentinel-derived HLS), not a combined Landsat/Sentinel array. Existing GeoParquet means we do not first harvest this collection through CMR STAC API queries. Metadata preparation and access still have costs. Keep auth, partition-access setup, and grid inspection in the linked notebook.

**Sources:** `../../docs/notebooks/hls-s3.ipynb`, `../../docs/notebooks/hls-https.ipynb`.

### 6. Open the big array — 5:45–8:00

**On screen:** two-column layout: the HLS `lazycogs.open(...)` call on the left and its actual Xarray HTML DataArray repr on the right, both visible together. Keep the explicit 2025 predicate, `EPSG:32615`, 30 m resolution, daily grouping, and RGB bands legible. Demonstrate flexibility with a brief toggle between two runnable, verified configurations of the same archive: the baseline daily view and a monthly view with a supported scene-cloud-cover predicate. Highlight the changed filter/grouping arguments; verify the relevant field exists in the HLS index before selecting the predicate. Settle on the daily configuration and its actual repr for the downstream examples; a second full repr is unnecessary. State that the alternate configuration requires another open, not mutation of the existing array. Use named, preconfigured variables for the archive, aligned extent, client, and store; do not expand their setup. Preserve the real HTML repr and check readability within the right column rather than replacing it with a hand-built summary. Link the full notebooks.

**Visible footnote:** “Direct HLS S3 access requires compute in AWS us-west-2. Use HTTPS elsewhere.” Link the HTTPS notebook.

**Speaker notes:** Let the HTML repr show dimensions, dtype, coordinates, and logical size. Explain that this size describes the array if materialized, not downloaded bytes or resident memory. Opening queries a representative item, inspects COG metadata per selected band, queries timestamps, and builds coordinates. No pixel windows yet. Avoid unconditional “instant” or “no files touched” claims. Mention Dask chunks where needed for lazy downstream arithmetic rather than promising that backend-lazy indexing makes all operations lazy.

**Sources:** use `../../docs/notebooks/hls-s3.ipynb` as the reference for the explicit 2025 predicate and caller-supplied hive-aware DuckDB client. The HTTPS notebook currently lacks an explicit year predicate and does not pass its configured client in the open call; reconcile these before capturing a year-specific result.

### 7. Follow one chunk — 8:00–10:15

**On screen:** a short `.sel(...)` expression above one read-path diagram; reveal `.compute()` to trigger the sequence in stages:

selection → chunk footprint/time predicate → GeoParquet query → matching COGs → overview/window reads → reprojection + mosaic → array result

Keep a small map alongside the diagram: highlight the requested window, then reveal the intersecting source footprints. Name libraries at their stages: rustac + DuckDB for asset discovery; async-geotiff + obstore for COG reads; pyproj + NumPy for reprojection and mosaicking. Keep these labels secondary, not a separate dependency list. If the slide becomes too busy, simplify or move labels into speaker notes rather than shrink the main diagram.

**Speaker notes:** rustac + DuckDB handle item queries; async-geotiff + obstore handle COG access; pyproj + NumPy handle reprojection and mosaicking. Xarray supplies indexing; Dask provides optional chunk scheduling. Queries run per requested time step and spatial chunk. A point request still reads compressed COG tiles, not a single pixel's worth of network bytes. Async I/O overlaps reads within a chunk.

Keep this slide focused on one requested chunk at the HLS array's existing 30 m grid. Name overview/window selection as a read stage, but save the resolution-pyramid explanation for the Southwest example's 300 m target grid. A `.sel(...)` operation does not change the array resolution.

**Source:** `../../ARCHITECTURE.md`.

### 8. Inspect the request — 10:15–12:15

**On screen:** keep plan inspection as a dedicated slide, separate from the preceding read-path slide. Use a two-column layout: the small HLS `.sel(...)` expression and `subset.lazycogs.explain()` on the left; a cropped, real plan summary for that selection on the right. Highlight matching-item and planned-COG-read counts using the actual output labels. Keep the selected footprint visible in a small map inset. Leave optional header inspection out of the main example.

**Speaker notes:** Connect the subset footprint to matching items and planned COG reads. Explain queries the index but does not read pixels. `fetch_headers=True` adds COG header inspection and overview/window information; use a prepared example if helpful. Planned COG reads are not exact HTTP request or byte counts.

**Sources:** adapt the plan-inspection pattern from `../../docs/notebooks/explain.ipynb` to the HLS example.

### 9. Compute the subset — 12:15–15:00

**On screen:** show both results in sequence: first a short HLS point selection across the year with a plot of the RGB-band time series; then reveal a small spatial selection with RGB images for a few dates. Use short, verified selections from the HLS notebook followed by `.compute()`. Keep the point and spatial window marked on the HLS coverage map to relate both requests to the full array extent. Use actual Xarray HTML reprs for intermediate DataArrays only if helpful, and plots for pixel results. No timing claim is needed.

**Speaker notes:** Contrast the small request with the year-long, zone-scale logical array. Choose a point with an interpretable seasonal pattern; label the plotted band's actual value units/scaling. Use nearby RGB windows at selected dates to explain the same pattern, clouds, and gaps, rather than presenting unrelated results. Preserve missing observations and make clear that cloud-filtered scenes are not pixel-cloud-masked data. Keep the selected point/window visible on the map so the audience can relate it to the read path. Report timings only if captured and documented for this run; this slide does not need a benchmark to make its point.

**Sources:** point and spatial selections in `../../docs/notebooks/hls-s3.ipynb` and `../../docs/notebooks/hls-https.ipynb`.

### 10. Many source projections, one Southwest array — 15:00–17:45

**On screen:** open with Southwest source footprints colored by verified source CRS as a dedicated reveal, then transition to a two-column layout: compact `lazycogs.open(...)` code on the left and the actual Xarray HTML repr for the monthly array on the right. Reveal the seasonal RGB mosaics from slide 2 as the result, connecting back to the opening imagery. Highlight the single target grid (`EPSG:3310`, 300 m) and:

```python
time_period="P1M",
filter="eo:cloud_cover < 50",
sortby="eo:cloud_cover",
```

**Overview reveal:** use the explicit 300 m target grid to introduce resolution-aware COG reads with a small resolution pyramid. Use actual source-band resolutions and overview levels verified from the example assets. The implementation chooses the coarsest available overview no coarser than the target resolution in source CRS, falling back to full resolution when needed, then samples onto the target grid. Explain that this is a new coarser-grid array definition, not an automatic optimization of arbitrary downstream Xarray `.coarsen()` operations. Existing overview pixels may have been generated with a different resampler; lazycogs' own sampling is nearest-neighbor. Do not add a second unrelated resolution example.

**Speaker notes:** Name the product explicitly: monthly first-valid mosaics ordered by scene cloud cover, not monthly means or medians. Emphasize the contrast with native-grid HLS: Southwest source imagery spans different projections, but the user requests one aligned array in a chosen target CRS and resolution. Verify the source CRS distribution before labeling the map or stating a count. Explain first-valid-pixel mosaicking in source order. Scene-level cloud ranking is not a per-pixel cloud mask or a guarantee of cloud-free output. Omit STAC search, auth, and grid setup. Use staged reveals so code, HTML repr, and imagery remain readable within one slide.

**Source:** `../../docs/notebooks/demo_southwest_monthly.ipynb`.

### 11. The scope we chose — 17:45–19:15

**On screen:** reuse a simplified read-path diagram with two short boundary labels: “COGs + STAC only” and “Nearest-neighbor resampling only (for now)”. Pair these boundaries with the affirmative fit: “Flexible array views over STAC-indexed COG archives—not a general raster engine.” Make clear that it is not a one-size-fits-all tool.

**Speaker notes:** Connect the implementation back to the GTI packaging experience. We reuse narrowly scoped libraries and keep the Xarray interface, but own the integration, metadata contracts, reprojection, and mosaicking. GDAL-free does not mean GDAL feature parity. STAC input means a compatible GeoParquet index and supported asset/metadata contracts, not arbitrary STAC API or catalog input. The COG/STAC scope and current nearest-neighbor-only resampling are deliberate limits, not a claim to cover every raster workflow; “for now” is not a promised delivery date for other resamplers. Small spatial chunks can multiply index queries and COG opens; chunk sizing still matters. The original desire for distributed work is a motivation, not evidence that every worker deployment configuration works today.

**Sources:** `../../README.md`, `../../ARCHITECTURE.md`, `../../docs/guides/chunking.md`.

### 12. Try it on your archive / discussion — 19:15–25:00

**On screen:** an imagery result, `pip install lazycogs`, repository/docs links, and a QR code once URLs are finalized. Leave this slide visible during discussion.

**Speaker notes:** Close by 20:00: prepare an index, describe an array, select a window, inspect and compute. Invite questions about archive layouts and analysis needs. Reserve the remaining 5 minutes for discussion.

## Backup material (outside the main deck)

- **HLS NDVI:** red/NIR and separate Fmask QA data, masking, then a monthly maximum over a subset. Validate dtype/nodata and QA behavior. Use Dask for lazy arithmetic. Source: `../../docs/notebooks/hls-analysis.ipynb`. Keep authentication out of the main talk.
- **Comparison with odc-stac:** existing notebook as a starting point, not proof of a general speedup. Record versions, hardware, network/cache state, item counts, grouping semantics, and chunk strategies; verify comparable outputs.
- **Native-grid alignment:** `align_bbox()` and source CRS choices from the quickstart.
- **Async workflows:** `.load_async()` and low-level `read_chunk_async()`; distinguish async I/O from Dask scheduling.
- **Storage/auth and temporal grouping:** public Sentinel-2 versus Earthdata access; daily/monthly/exact timestamps and overlap semantics.

## Preparation

- Verify the two alternate HLS open configurations, including availability and semantics of the chosen scene-cloud-cover field. Capture the baseline daily repr used downstream; do not claim flexibility using unrun snippets.
- Select a scientifically interpretable HLS point and nearby RGB window; document value units/scaling and preserve gaps. Verify actual Southwest source overview levels for the 300 m example.
- Rehearse the four opening slides to 5 minutes, with the HLS map limited to roughly 45 seconds. Prioritize readable reveals over animation complexity; preserve time for explanation and questions.
- Extract a short, working GTI example from the prototype; avoid unverified minimum-version or current-distribution claims.
- Capture the HLS year-long, zone-scale array's real HTML repr, footprint, subset plan, point result, and RGB images from one reproducible run. Verify temporal scope, collection, dimensions, logical size, and any timings shown; do not invent a petabyte/terabyte headline.
- Capture the Southwest open call, real HTML repr, source-CRS footprint map, and composite. Verify which source projections occur. Use static results as the talk's baseline; a live demo should be optional.
- Preserve Xarray's HTML representation and styling in the presentation assets. Do not substitute terminal output or manually reconstruct its metadata display. Check readability and offline rendering when integrating the assets into Slidev.
- Rerun selected examples against the current package. Some quickstart prose is stale: opening now inspects representative COG metadata.
- Keep narrative detail and caveats in speaker notes, not in visible bullet lists. Rehearse the background to a 5-minute cutoff and the presentation to a 20-minute cutoff.
- Preserve the submitted proposal in `AGENTS.md`.
