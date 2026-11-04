---
title: "lazycogs: super lazy analysis of massive COG archives"
info: |
  FOSS4GNA 2026 · Sacramento, CA · 2026-11-03
  Henry Rodman · Development Seed
highlighter: shiki
colorSchema: light
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

<img :src="lazycogsLogo" alt="lazycogs" class="title-wordmark" />

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
    <p text-3xl font-300>FOSS4GNA 2026<br>Sacramento, CA</p>
    <h4 text-md font-mono>2026-11-03</h4>
    <h5 text-sm><code text-primary>Henry Rodman · Development Seed</code></h5>
  </div>
</DecorativeRectangle>

<!--
0:00–0:20. Henry Rodman, Development Seed. “Query satellite imagery collections like everyone wants to: as a big array.” This is a view over an existing archive, not a proposal to rewrite all its pixels. Target 20 minutes presenting, then 5 minutes discussing.
-->

---
layout: default
class: image-slide
title: I want to work with this as an array
---

# I want to work with this as an array.

<AssetImage asset="southwest-notebook/mosaic-2025-05.png" alt="Actual May 2025 Southwest Sentinel-2 RGB mosaic plotted in the monthly notebook" class="hero-image" />

<p class="caption">May 2025 · Sentinel-2 · Southwest US · 300 m</p>

<!--
0:20–1:10. Start with the actual saved May plot from docs/notebooks/demo_southwest_monthly.ipynb. The notebook covers March–June 2025, not the old deck's March/July/November sequence. This image is assembled from overlapping observations, not one ready-made image in the archive. Observations are acquired at different times and contribute unevenly. Later we will show exactly how this monthly view was defined and selected. Display stretch is stored DN 0–4000; not a physical reflectance claim. Source provenance is in assets/southwest-notebook/.
-->

---
layout: default
class: index-slide
title: GeoParquet holds metadata. COGs hold pixels.
---

# GeoParquet holds metadata. COGs hold pixels.

<p class="lead">STAC items: footprints · timestamps · properties · asset links</p>

<div class="index-content">
<div>

```python
from rustac import DuckdbClient

index_uri = (
    "https://s3.us-west-2.amazonaws.com/nasa-maap-data-store/"
    "file-staging/nasa-map/hls-stac-geoparquet-archive/v2/"
    "HLSS30_2.0/year=2025/month=1/HLSS30_2.0-2025-1.parquet"
)
items = DuckdbClient().search(
    index_uri,
    bbox=(-96, 44, -90, 48),
    datetime="2025-01-01/2025-01-31",
)
```

</div>
<div class="index-link">
<AssetQRCode value="https://maap-project.github.io/hls-stac-geoparquet-archive/" />
<a href="https://maap-project.github.io/hls-stac-geoparquet-archive/">maap-project.github.io/<br>hls-stac-geoparquet-archive/</a>
</div>
</div>

<p class="caption">NASA MAAP publishes the HLS index. Someone prepares and maintains it.</p>

<!--
1:10–2:30. Introduce STAC at rest: items in GeoParquet describe the imagery; their assets link to COG pixels elsewhere. rustac's DuckdbClient searches that metadata. The URI is a concrete January 2025 HLSS30_2.0 object, using the archive's unpadded month key. The existing capture resolved the current Iceberg snapshot and verified this exact public object over HTTPS. This short example is a January regional item search, not the full-year array we open later.

Do not imply a remote **/*.parquet wildcard can be searched anonymously. Exact public objects are readable but bucket listing is denied. The S3 notebook configures AWS credentials with ListBucket privileges for wildcard searches; the HTTPS notebook copies individual objects to a local hive tree. Our verified 2025 capture follows the published Iceberg metadata and manifests, then caches the twelve 2025 HLSS30 partitions locally. See assets/hls/reproduce.py and manifest.json. Index preparation and metadata access still cost work. NASA MAAP maintains the index; lazycogs does not decide which scenes we want.
-->

---
layout: default
title: The useful array interface already exists
---

# The useful array interface already exists

```python
import odc.stac

cube = odc.stac.load(
    items,
    bands=["B04", "B03", "B02"],
    crs="EPSG:32615",
    resolution=30,
    chunks={"time": 1, "x": 2048, "y": 2048},
)
small = cube.sel(
    x=slice(574000, 579000),
    y=slice(5199000, 5194000),
).compute()
```

<p class="sequence">retrieve items → build lazy array / task graph → do the work</p>
<p class="caption">odc-stac · stackstac</p>

<!--
2:30–3:45. Credit odc-stac and stackstac: these tools established the useful array interface and work well for many satellite workflows. `items` is the in-memory STAC item collection from the preceding search; stackstac.stack(items, ...) follows the same broad input pattern. This is a conventional example, not a timed comparison or a captured raster result.

Pixel reads are lazy. Dask graph optimization can prune work for a later selection. The motivation is the up-front item enumeration and graph construction for a large view when the eventual request may be small, not a claim that either tool eagerly downloads every pixel. In an earlier large-scale project, very large task graphs and repeated STAC API searches were obstacles in the speaker's workflow. Do not generalize that experience into a universal scaling limit or claim lazycogs is a drop-in replacement.
-->

---
layout: center
title: What if we also deferred finding the files?
---

# We already defer reading pixels.
# What if we also deferred finding the files?

<p class="sequence spacious">define array → select → query matching assets → read pixels</p>

<p class="lead">Carry query criteria, not the complete item collection.</p>

<!--
3:45–4:45. This is the conceptual contrast: define the view first; when a spatial chunk and time step are requested, query the index for assets relevant to that request. Carry criteria instead of retaining the entire matching item collection up front.

Not zero work at open. lazycogs queries representative items, inspects COG headers, queries timestamps, and creates coordinates. It still needs a prepared index. Nor does it eliminate Dask graphs: Dask is available for chunk scheduling and downstream operations. A change of supported metadata filters, temporal grouping, extent, or target grid defines a new array. .sel(...) selects within an existing view; it does not redefine those choices.
-->

---
layout: default
title: GTI worked. Packaging was the obstacle.
---

# GTI worked. Packaging was the obstacle.

```python
import rioxarray

red = rioxarray.open_rasterio(
    "/tmp/tmp0umyk8_2/items_projected.parquet",
    driver="GTI",
    LOCATION_FIELD="assets.red.href",
    FILTER="datetime >= '2025-06-01' AND datetime < '2025-06-02'",
    RESX=10, RESY=10,
    chunks={"x": 2048, "y": 2048},
)
```

<p class="sequence">GDAL + GTI + Parquet → rasterio → rioxarray</p>

[github.com/hrodmn/stac-gti-xarray](https://github.com/hrodmn/stac-gti-xarray)

<!--
4:45–6:00. Read the actual local demo.ipynb and Dockerfile, not the absent src/stac_gti_xarray/_core.py. This snippet is the red-band, one-day core of the working experiment. The concrete projected Parquet path is derived from the saved run's /tmp/tmp0umyk8_2/items.parquet path and its items_projected.parquet assignment. It is a historical temporary path, not a persistent file shipped with the deck. Regenerating the demo creates another temporary directory. The displayed half-open day predicate is a simplified single-day filter; the prototype's loop uses a <= following-midnight bound.

LOCATION_FIELD selects the red asset's href column. The Parquet contains metadata and may contain all the other band asset columns too; it does not contain the raster pixels. GTI does source selection, reprojection, and mosaicking. Python loops over dates and bands and concatenates the arrays. The prototype uses per-band symlinks to avoid an opener-option caching issue; omit that workaround from the main snippet, which opens just red. The notebook also specifies the target bounds and other options in its full call.

It worked. The obstacle was distributing the native build chain and runtime linkage. The speaker originally built GDAL with Parquet support, then built rasterio against it. The current Dockerfile starts from ghcr.io/osgeo/gdal:alpine-normal-latest and uses uv --no-binary-package rasterio. Do not claim GTI failed at raster work, or that these historical packaging issues describe every current GDAL distribution. That experience motivated a deliberately narrower GDAL-free implementation.
-->

---
layout: default
class: array-slide
title: lazycogs — a year of HLS as an array
---

# lazycogs: a year of HLS as an array

<div class="array-columns">
<div>

```python
hls = lazycogs.open(
    "data/hls-stac-geoparquet-archive/"
    "v2/HLSS30_2.0/**/*.parquet",
    bbox=(166020, 0, 834000, 9329010),
    crs="EPSG:32615",
    resolution=30,
    time_period="P1D",
    bands=["B04", "B03", "B02"],
    datetime="2025-01-01/2025-12-31",
    filter="proj:epsg = '32615'",
    duckdb_client=client,
    store=hls_store,
)
hls
```

</div>
<div class="array-output">
<AssetFrame asset="hls/array.html" title="Actual verified 2025 HLS Xarray HTML representation" />
</div>
</div>

<p class="caption">≈15 TB logical array size. Not downloaded bytes or memory use.</p>
<p class="footnote">Direct HLS S3 access requires compute in AWS us-west-2. <a href="https://developmentseed.org/lazycogs/notebooks/hls-https/">HTTPS is the alternative</a>.</p>

<!--
6:00–8:30. This is the verified 2025 HLSS30_2.0 daily RGB view: band=3, time=365, y=310967, x=22266, int16, nodata=-9999. It is Sentinel-derived HLS, not a combined Landsat/Sentinel array. The literal bbox is aligned to the source grid in UTM zone 15N. About 15 TB is the size of the logical dense array if materialized, not bytes transferred or RAM allocated.

The actual HTML repr is assets/hls/array.html from the separately verified 2025 capture, not the stale saved HTTPS notebook output (494 time steps), or the S3 notebook's old ascending-y output. The displayed code shortens the equivalent CQL2 projection and datetime predicates; its full setup is in assets/hls/capture_assets.py. Run from docs/notebooks: the local hive tree contains the 12 manifest-listed 2025 partitions prepared by assets/hls/reproduce.py. client = DuckdbClient(use_hive_partitioning=True). hls_store is the preconfigured Earthdata-authenticated HTTPSStore from obtain_store(); the capture used HTTPS, not direct S3. HLS pixel access needs NASA Earthdata credentials. No credentials, downloads, or network calls are required to present these saved outputs.

Opening is not free: representative-item queries, COG headers for selected bands, timestamp queries, and coordinate construction happen up front. Pixel windows wait for a request. The coordinate convention is x ascending, y descending. The complete S3 and HTTPS workflows are linked from the docs. No opening timing or benchmark claim.
-->

---
layout: default
title: Load one point
---

# Load one point

```python
point = hls.sel(x=576465, y=5196159, method="nearest").compute()
```

<AssetImage asset="hls/point-rgb.png" alt="Verified 2025 daily HLS RGB values at the selected point, with missing dates retained as gaps" class="point-plot" />

<p class="caption">EPSG:32615 · stored digital numbers · missing observations remain gaps</p>

<!--
8:30–10:30. Real plot from the unfiltered-by-cloud daily baseline just shown. The point is at x=576465, y=5196159 in EPSG:32615. The verified run has 123 days with all RGB bands valid and 242 missing dates; nodata remains gaps, not a line bridged across absent observations. Use the original PNG, not the later SVG's simplified-path marker additions. Values are stored DNs; no physical reflectance scale or units were verified in the index. Clouds can still produce bright values in this unfiltered view.

A point request still reads compressed COG tiles; it is not one pixel's worth of network bytes. The compute path is query index → find COGs → read windows/overviews → reproject/mosaic. Queries happen for requested spatial chunks and time steps. No need to read the entire year-long spatial array. No transfer-size or timing claim is being made.
-->

---
layout: default
class: window-slide
title: A new scene filter, then a small window
---

# A new scene filter, then a small window

```python
# hls_args: same index, bbox, grid, year, bands, client and store
filtered = lazycogs.open(
    **hls_args,
    filter="proj:epsg = '32615' AND eo:cloud_cover < 30",
)
nearby = filtered.sel(
    x=slice(573965, 578965), y=slice(5198659, 5193659),
    time=["2025-04-07", "2025-07-14", "2025-10-10"],
).compute()
```

<AssetImage asset="hls/nearby-rgb.png" alt="Verified 5 km HLS RGB windows for April 7, July 14 and October 10, from a separately opened scene-cloud-filtered daily array" class="window-images" />

<p class="caption">5 km window · scene cloud cover &lt;30% ≠ pixel cloud masking</p>

<!--
10:30–12:45. This montage is not an output of the unfiltered baseline. Explicitly reopen with a supported scene-cloud predicate. For this short snippet, hls_args is the previous call's arguments including time_period='P1D', with its old filter removed. It does not include a chunks or sortby override. The capture script uses the equivalent CQL2-JSON and full-day timestamp range; see assets/hls/capture_nearby_rgb.py. A code/configuration crosswalk is in the deck outline.

These are the exact 5 km bounds around the point. Candidate observations were restricted to scene cloud <30 and dates chosen nearest April 15, July 15, October 10. Actual dates are April 7 (4% scene cloud), July 14 (6%), October 10 (0%). Plot stretch is DN 0–2000. This is an illustrative seasonal scene sequence, not quantitative phenology. Scene cloud filtering is not a pixel mask or guarantee of clear pixels.

Optional: before computing, nearby.lazycogs.explain() would query the same index without reading pixels. The captured three-date plan has 9 band/chunk reads and 15 planned COG reads; do not equate these with HTTP request counts. Avoid a dedicated plan diagram. Supported metadata filters, grouping, extent, and resolution are the user's choices. Changing those defines a new array; .sel(...) selects inside it.
-->

---
layout: default
title: This time, prepare our own index
---

# This time, prepare our own index

```python
import rustac

await rustac.search_to(
    "data/ca_spring_2025.parquet",
    href="https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-c1-l2a"],
    datetime="2025-03-01/2025-06-30",
    bbox=(
        -125.38223191231621, 32.29621234930089,
        -111.75677519150827, 42.51937087572006,
    ),
)
```

<p class="sequence">STAC API search → local STAC GeoParquet → array views</p>

<!--
12:45–14:15. Return to Southwest Sentinel-2. Unlike the existing published HLS archive, we prepare this index ourselves. This is the search_to workflow from demo_southwest_monthly.ipynb, with the geographic bbox expanded to literal values. Verified with pyproj.Transformer.from_crs(3310, 4326, always_xy=True).transform_bounds(-444000, -609000, 681000, 500000). The notebook caches this file and searches only if it does not exist; limit=100 is the search page size and was omitted here for brevity, not a 100-item cap. March–June 2025 only.

The metadata is now local; COG asset links still point to the pixels. Index preparation is deliberately separate from opening a view and computing a subset. No live API search is necessary for the talk. These are compatible STAC-indexed COGs, not arbitrary raster inputs. Earth Search's public Sentinel-2 assets use an unsigned store in this notebook. Keep this setup out of the visible code on the following slide.
-->

---
layout: default
class: array-slide
clicks: 1
title: Your scene selection. Your target grid.
---

# Your scene selection. Your target grid.

<div v-if="$clicks === 0" class="array-columns">
<div>

```python
monthly = lazycogs.open(
    "data/ca_spring_2025.parquet",
    crs="EPSG:3310",
    bbox=(-444000, -609000, 681000, 500000),
    resolution=300,
    time_period="P1M",
    bands=["red", "green", "blue"],
    dtype="int16",
    filter="eo:cloud_cover < 50",
    sortby="eo:cloud_cover",
    store=store,
)
monthly
```

</div>
<div class="array-output">
<AssetFrame asset="southwest-notebook/array.html" title="Actual March–June 2025 Southwest monthly Xarray HTML representation from the notebook" />
</div>
</div>

<div v-else class="may-result">
<div>

```python
may = monthly.chunk(x=1024, y=1024).sel(time="2025-05-01")
await may.load_async()
```

</div>
<AssetImage asset="southwest-notebook/mosaic-2025-05.png" alt="The same May 2025 notebook RGB image shown at the opening" class="may-image" />
</div>

<p class="caption">Monthly first-valid mosaic, ordered by scene cloud cover · nearest-neighbor sampling</p>

<!--
14:15–18:45. Start with the actual notebook Xarray HTML repr and its precise monthly open. Four time coordinates cover March–June; band=3, time=4, y=3697, x=3750, int16, about 333 MB logical size. This is not the old full-year uint16 seasonal preview configuration. Only variable names and uppercase CRS spelling are adapted for the slides. store = lazycogs.store_for('data/ca_spring_2025.parquet', skip_signature=True), as in the notebook. Notes and prepared output hide setup, not semantics.

Emphasize: “The index does not decide which scenes you want. You do.” We choose supported metadata filters, grouping, extent, target CRS and resolution. There is no single imposed scene selection or precomputed mosaic. A different filter, temporal grouping, or resolution defines another array; .sel(...) selects inside this view. Do not imply arbitrary query syntax or arbitrary pixel operations are supported.

One simple reveal replaces the open/repr with the May selection, async load, and the exact opening image. No animation or custom diagram. The notebook chunks to 1024x1024 before selecting May, loads with await load_async(), then plots RGB with vmin=0, vmax=4000. This saved plot is the actual 300 m result, not a separately reopened coarser preview.

Multiple source projections become one requested EPSG:3310 target grid. Appropriate source COG overviews support the coarser 300 m output; this is not an automatic rewrite of an arbitrary downstream .coarsen operation. lazycogs currently samples nearest-neighbor. Existing overview pixels may have been produced with another resampler. Monthly output is first-valid mosaicking in ascending scene-cloud order, not a mean, median, or pixel cloud mask. Different scenes can supply different parts of the output. Scene cloud cover is a scene-level property, not a clear-pixel guarantee.
-->

---
layout: default
class: closing-slide
title: lazycogs — discussion
---

<script setup>
import closingLogo from '../../logo-text.svg?url'
</script>

<img :src="closingLogo" alt="lazycogs" class="closing-wordmark" />

<p class="closing-message">The index does not decide which scenes you want.<br><strong>You do.</strong></p>

<div class="closing-links">
<div>
<a class="docs-link" href="https://developmentseed.org/lazycogs/">developmentseed.org/lazycogs/</a>
<p><code>pip install lazycogs</code></p>
<a href="https://github.com/developmentseed/lazycogs">github.com/developmentseed/lazycogs</a>
</div>
<AssetQRCode value="https://developmentseed.org/lazycogs/" />
</div>

<!--
18:45–20:00, then leave this slide visible for 5 minutes of discussion. The deliberate scope is compatible STAC-indexed COGs, not a general raster engine. A GDAL-free implementation is a narrow integration choice, not GDAL feature parity. We reuse rustac/DuckDB for discovery, async-geotiff/obstore for COG reads, pyproj/NumPy for alignment, and Xarray with optional Dask scheduling. We still own the integration and metadata contracts; chunk sizes and index layout matter. Distributed chunkwise analysis motivated this work, not a guarantee that every worker deployment works unchanged.

The docs URL is verified against mkdocs.yml site_url and the docs workflow: https://developmentseed.org/lazycogs/. The static local QR encodes that same URL. All talk images and HTML outputs are bundled locally; links are optional follow-up resources. Invite questions about which archive and array view the audience wants to work with.
-->
