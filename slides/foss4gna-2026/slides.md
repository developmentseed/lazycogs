---
title: "lazycogs: super lazy analysis of massive COG archives"
info: |
  FOSS4GNA 2026 · Sacramento, CA · 2026-11-03
  Henry Rodman · Development Seed
highlighter: shiki
fonts:
  webfonts: []
favicon: /images/logos/hor--pos@2x.png
drawings:
  persist: false
theme: ../theme
layout: title
---

<script setup>
import lazycogsLogo from '../../logo-text.svg?url'
</script>

<img :src="lazycogsLogo" alt="lazycogs relaxed squares logo with wordmark" class="title-wordmark" />

::subtitle::
super lazy analysis of massive COG archives

<DecorativeRectangle
  width="50%"
  height="40%"
  zIndex=20
  :position="{ bottom: '2%', right: '2%' }"
  :customStyle="{ mixBlendMode: 'multiply' }"
>
  <div w-full h-full relative flex flex-col items-end justify-end p-4 text-white text-right>
    <p text-3xl font-300>
      FOSS4GNA 2026 <br>
      Sacramento, CA
    </p>
    <h4 text-md font-mono>2026-11-03</h4>
    <h5 text-sm><code text-primary>Henry Rodman</code></h5>
  </div>
</DecorativeRectangle>

<!--
Introduce Development Seed, then say: “Query satellite imagery collections like everyone wants to: as a big array.” The array is a view over existing COGs, not a requirement to rewrite the archive.
-->

---
layout: none
clicks: 3
title: From observations to a mosaic
---

<SouthwestStoryboard :step="$clicks" />

<!--
0:30–1:30. Start with the grand monthly mosaic. Rotate about the vertical axis and separate three verified seasonally spaced mosaics. Reveal one representative source datatake from each month as transparent pass layers with acquisition dates and footprint outlines, then reassemble the mosaics. The animation explains sources becoming a mosaic, not the provenance of every output pixel. The archive contains observations from different times with overlapping footprints; each monthly mosaic summarizes observations within that month, and scenes do not contribute equally. A Sentinel-2 granule is a tile, not a full orbit swath. The displayed pass selection is illustrative, not exhaustive. Scene-level cloud ranking does not guarantee cloud-free pixels. Reuse these exact images on slide 10.
-->

---
layout: default
clicks: 1
title: Define the view. Discover when needed.
---

<div class="interface-slide">
  <h1>Define the view. Discover when needed.</h1>
  <div class="view-axes">
    <div class="view-card"><small>Existing STAC GeoParquet</small><strong>daily · all observations</strong><span>time / y / x</span></div>
    <div class="view-arrow">→</div>
    <div class="view-card"><small>New view definition</small><strong>monthly · scene filter</strong><span>first valid, cloud-cover order</span></div>
  </div>
  <div class="selection-flow">
    <pre><code>subset = imagery.sel(
    time=slice("2025-03", "2025-06"),
    x=slice(xmin, xmax),
    y=slice(ymax, ymin),
)</code></pre>
    <span class="selection-arrow">→</span>
    <div class="window-card">small window<br><span>array view</span></div>
  </div>
  <div v-click class="discover-flow">
    <span>prepare index</span><b>→</b><span>open array</span><b>→</b><span>compute selection</span><b>→</b><span>query matching assets</span><b>→</b><span>read pixels</span>
  </div>
</div>

<!--
1:30–3:15. Different applications need different filters, temporal grouping, target grids, and extents. The array view should be easy to define without preparing one product or fetching pixels prematurely. This is a design goal, not an unconditional speed guarantee. Scene cloud-cover filtering is not a pixel cloud mask. Credit stackstac and odc-stac for establishing the array pattern. In one large-scale project, task graphs and repeated STAC API queries became obstacles; describe that experience, not universal library limits. Chunkwise distributed processing motivated the work but is not a deployment guarantee.

The xarray selection is familiar, but the view is defined separately. A descending y slice assumes a north-up grid. Someone must prepare or publish searchable metadata first: HLS uses an existing index; the Southwest workflow prepares one from a STAC search. The benefit is avoiding enumeration and retention of the entire matching item collection for each view, not eliminating index preparation. Opening inspects representative COG metadata and timestamps; it does not read pixel windows. Changing filters, grouping, or target grid requires another open. Do not suggest that lazycogs eliminates Dask graphs.
-->

---
layout: default
title: GTI worked. Packaging was the obstacle.
---

# GTI worked. Packaging was the obstacle.

<div class="gti-layout">
  <div>
    <p class="eyebrow">A working open path</p>
    <pre><code>rioxarray.open_rasterio(
    "GTI:/path/to/stac.parquet",
    LOCATION_FIELD="assets.red.href",
    FILTER="datetime &gt;= '2025-03-01 00:00:00' AND datetime &lt; '2025-03-02 00:00:00'",
)</code></pre>
    <a href="https://github.com/hrodmn/stac-gti-xarray" target="_blank">github.com/hrodmn/stac-gti-xarray</a>
  </div>
  <div class="build-chain">
    <div>GDAL<br><small>GTI + Parquet</small></div><b>→</b>
    <div>rasterio<br><small>built against it</small></div><b>→</b>
    <div>rioxarray</div>
    <p>Native build chain + runtime setup</p>
  </div>
</div>

<!--
3:15–5:00. The experiment worked: GTI supplied spatial source selection, reprojection, and mosaicking; the Python layer assembled dates and bands. SQL filtering through an open option was useful. You built GDAL from source with Parquet/GeoParquet support, then built rasterio against that installation. Distributing the native build chain and runtime setup was the obstacle. The current prototype Dockerfile instead starts from a GDAL image and compiles rasterio against it. This motivated a tightly scoped GDAL-free architecture, not a claim that GTI failed at raster work or a general claim about current GDAL packaging.
-->

---
layout: none
title: A year of HLS, an entire UTM zone
---

<div class="map-first">
  <AssetImage asset="hls/coverage.svg" alt="Observed 2025 HLS item footprints in UTM zone 15N with selected point and window" />
  <div class="map-caption"><strong>2025 · 30 m · RGB</strong><span>UTM zone 15N · observed item footprints · HLSS30_2.0</span></div>
</div>

<!--
5:00–5:45. The reference is NASA MAAP's existing HLS STAC GeoParquet archive. This example selects HLSS30_2.0 (Sentinel-derived HLS), not a combined Landsat/Sentinel array. The archive avoids first harvesting this collection through CMR STAC API queries, but metadata preparation and access still have costs. Keep auth, partition-access setup, and grid inspection in the linked notebook.
-->

---
layout: default
title: Open the big array
---

# Open the big array

<div class="array-layout">
  <div class="code-pane">
    <pre><code>imagery = lazycogs.open(
    archive,
    datetime="2025-01-01T00:00:00Z/2025-12-31T23:59:59Z",
    filter=zone15,
    crs="EPSG:32615",
    resolution=30,
    time_period="P1D",
    bands=["B04", "B03", "B02"],
)
// Alternate filters or grouping require another open</code></pre>
    <div class="tiny-links"><a href="https://developmentseed.org/lazycogs/notebooks/hls-s3/" target="_blank">S3 notebook</a> · <a href="https://developmentseed.org/lazycogs/notebooks/hls-https/" target="_blank">HTTPS notebook</a></div>
  </div>
  <div class="repr-pane"><AssetFrame asset="hls/array.html" title="Actual Xarray DataArray HTML representation" /></div>
</div>
<p class="footnote">Direct HLS S3 access requires compute in AWS us-west-2. Use HTTPS elsewhere.</p>

<!--
5:45–8:00. Use named preconfigured archive, extent, client, and store. Keep the baseline explicit 2025, EPSG:32615, 30 m, daily, RGB. The actual array HTML is a saved historical notebook result, not a fresh run. The current MAAP parquet request returned 403, so the HLS `eo:cloud_cover` predicate and alternate monthly open are not verified; do not present the candidate as runnable or supported. The repr's dimensions, dtype, coordinates, and logical size are real. Logical size describes materialized array size, not downloaded bytes or resident memory. Opening queries a representative item, inspects COG metadata per selected band, queries timestamps, and builds coordinates. No pixel windows are read yet. Avoid “instant” and “no files touched.” Dask chunks matter for lazy downstream arithmetic. The HTTPS notebook may lack year scoping; use only a verified 2025 result. Direct HLS S3 requires AWS us-west-2; HTTPS is the alternative.
-->

---
layout: default
clicks: 3
title: Follow one chunk
---

# Follow one chunk

<div class="chunk-layout">
  <div class="chunk-steps">
    <pre><code>window = imagery.sel(
    time=chosen_date,
    x=slice(xmin, xmax),
    y=slice(ymax, ymin),
)
<span v-click="1">result = window.compute()</span></code></pre>
    <div class="step-row"><i>1</i><span>selection defines chunk</span></div>
    <div class="step-row" v-click="1"><i>2</i><span>GeoParquet query per time + chunk</span><small>rustac · DuckDB</small></div>
    <div class="step-row" v-click="2"><i>3</i><span>matching COGs → overview/window reads</span><small>async-geotiff · obstore</small></div>
    <div class="step-row" v-click="3"><i>4</i><span>reproject + mosaic → array</span><small>pyproj · NumPy</small></div>
  </div>
  <div class="map-inset"><AssetImage asset="hls/selection.svg" alt="Observed HLS 2025 footprints with selected window" /><div class="map-overlay">2025 footprint coverage · selected window</div></div>
</div>

<!--
8:00–10:15. Keep the request on the existing 30 m grid. Xarray supplies indexing; Dask provides optional chunk scheduling. rustac + DuckDB query items; async-geotiff + obstore read COGs; pyproj + NumPy reproject and mosaic. Queries run per requested time step and spatial chunk. A point request can still read compressed COG tiles, not a single pixel's worth of network bytes. Async I/O overlaps reads within a chunk. Save resolution-pyramid details for Southwest. .sel() does not change resolution.
-->

---
layout: default
title: Inspect the request
---

# Inspect the request

<div class="plan-layout">
  <div>
    <pre><code>window = imagery.sel(
    time=selected_dates,
    x=slice(point_x - 2500, point_x + 2500),
    y=slice(point_y + 2500, point_y - 2500),
)
plan = window.lazycogs.explain()</code></pre>
    <div class="map-small"><AssetImage asset="hls/selection.svg" alt="HLS 2025 item footprints and selected window for the explain plan" /></div>
  </div>
  <AssetImage asset="hls/plan.svg" alt="Actual lazycogs explain plan summary for the selected HLS request" />
</div>

<!--
10:15–12:15. Read the actual summary labels and counts. explain() queries the index but does not read pixels. fetch_headers=True adds COG header inspection and overview/window information, but keep it out of the visible main example. Planned COG reads are not exact HTTP request or byte counts.
-->

---
layout: default
clicks: 1
title: Compute the subset
---

# Compute the subset

<div class="results-layout">
  <div class="result-map"><AssetImage asset="hls/selection.svg" alt="HLS 2025 footprints marking the point and nearby window" /><span>point · nearby window</span></div>
  <div class="result-figures">
    <div><h3>Point RGB · 2025</h3><AssetImage asset="hls/point-rgb.svg" alt="Actual HLS RGB-band digital number series through 2025 with missing dates as gaps" /></div>
    <div v-click><h3>Nearby RGB · 3 selected dates</h3><AssetImage asset="hls/nearby-rgb.png" alt="Actual HLS RGB window on three selected dates" /></div>
  </div>
</div>

<!--
12:15–15:00. The available real result is a saved RGB plot for a 15 km by 10 km HLS window on ten September 2025 dates. Its values are raw digital numbers with a display stretch of 0–2000, not reflectance. Keep the requested grid and selected window visible. The requested full-year point time series is unavailable because the current archive metadata query returned 403; do not narrate this plot as a year-long point series. Preserve gaps and do not imply scene filtering is pixel cloud masking. No timing claim.
-->

---
layout: default
clicks: 2
title: Many projections. One Southwest array.
---

# Many projections. One Southwest array.

<div v-if="$clicks === 0" class="southwest-map-full">
  <AssetImage asset="southwest/footprints.svg" alt="Representative Southwest pass granule footprints colored by source CRS" />
  <span>selected pass footprints · CRS</span>
</div>
<div v-else class="southwest-layout">
  <div class="southwest-example">
    <pre><code>monthly = lazycogs.open(
    parquet,
    crs="EPSG:3310",
    resolution=300,
    time_period="P1M",
    filter="eo:cloud_cover &lt; 50",
    sortby="eo:cloud_cover",
)</code></pre>
    <AssetFrame asset="southwest/array.html" title="Actual Xarray DataArray representation for the Southwest monthly array" />
    <div v-click class="overview-label">300 m target grid · choose an available source overview</div>
  </div>
  <div class="southwest-mosaics"><AssetImage asset="southwest/mosaic-2025-03.png" alt="March 2025 RGB mosaic" /><AssetImage asset="southwest/mosaic-2025-07.png" alt="July 2025 RGB mosaic" /><AssetImage asset="southwest/mosaic-2025-11.png" alt="November 2025 RGB mosaic" /></div>
</div>

<!--
15:00–17:45. Start with the CRS map, then show the open call, actual HTML repr, and the same seasonal mosaics introduced on slide 2. The example is monthly first-valid mosaics ordered by scene cloud cover, not means or medians. Different source projections are aligned onto the requested EPSG:3310, 300 m grid. Scene ranking is not a pixel cloud mask. First-valid pixels follow source order. For the overview reveal, use the recorded source-band resolutions/overview levels: choose the coarsest available overview no coarser than target resolution in source CRS, falling back to full resolution; then sample onto the target grid. Existing overview pixels may use another resampler. lazycogs sampling is nearest-neighbor. This is a new coarser-grid array definition, not automatic optimization of arbitrary xarray coarsen operations.
-->

---
layout: default
title: The scope we chose
---

# The scope we chose

<div class="scope-diagram">
  <div class="scope-flow"><span>STAC GeoParquet</span><b>→</b><span>selected COG windows</span><b>→</b><span>aligned array chunks</span></div>
  <div class="scope-boundaries"><strong>COGs + STAC only</strong><strong>Nearest-neighbor resampling only (for now)</strong></div>
  <p>Flexible array views over STAC-indexed COG archives<br><b>not a general raster engine</b></p>
</div>

<!--
17:45–19:15. Connect the boundaries to the GTI packaging experience. lazycogs reuses narrowly scoped libraries and keeps the xarray interface, but owns integration, metadata contracts, reprojection, and mosaicking. GDAL-free does not mean GDAL feature parity. STAC input means a compatible GeoParquet index and supported asset/metadata contracts, not arbitrary STAC API/catalog input. “For now” is not a promised delivery date for other resamplers. Small spatial chunks multiply index queries and COG opens; chunk sizing matters. Distributed execution is a motivation, not evidence every worker setup works today. See README, ARCHITECTURE, and chunking guide.
-->

---
layout: default
title: Try it on your archive
---

# Try it on your archive

<div class="closing-layout">
  <AssetImage asset="southwest/mosaic-2025-07.png" alt="July 2025 Southwest monthly Sentinel-2 imagery" />
  <div class="closing-card">
    <code>pip install lazycogs</code>
    <a href="https://github.com/developmentseed/lazycogs" target="_blank">github.com/developmentseed/lazycogs</a>
    <a href="https://developmentseed.org/lazycogs/" target="_blank">developmentseed.org/lazycogs</a>
    <AssetQRCode value="https://developmentseed.org/lazycogs/" />
  </div>
</div>

<!--
19:15–20:00. Close by 20 minutes: prepare an index, describe an array, select a window, inspect and compute. Leave this slide up for discussion. Invite questions about archive layout and analysis needs. Five minutes remain for discussion.
-->
