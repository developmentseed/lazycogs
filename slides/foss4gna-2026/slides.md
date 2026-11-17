---
title: "lazycogs: super lazy analysis of massive COG archives"
info: |
  FOSS4GNA 2026 · Sacramento, CA · 2026-11-04
  Henry Rodman · Development Seed
highlighter: shiki
colorSchema: light
fonts:
  webfonts: []
favicon: /favicon.svg
drawings:
  persist: false
routerMode: hash
theme: ../theme
addons:
  - slidev-addon-qrcode
layout: title
image: /images/theme/lena-delta.jpg
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
    <h4 text-md font-mono>2026-11-04</h4>
    <h5 text-sm><code text-primary>Henry Rodman · Development Seed</code></h5>
  </div>
</DecorativeRectangle>
<LogoHorPos position="top-left" height="24px" />

<!--
0:00–0:20

- Henry Rodman, Development Seed.
- Query satellite imagery collections as a big array, without rewriting the archive.
- 20-minute talk, then 5 minutes for discussion.
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
0:20–1:10

- May 2025 Southwest Sentinel-2 mosaic from the monthly notebook.
- Overlapping observations from different times contribute to this image.
- We'll return to how we define and select this monthly view.
-->


---
layout: default
class: stac-pattern-slide
title: Search STAC, then hand the items to odc-stac
---

# Search STAC, then hand the items to xarray ✨

```python
from pystac_client import Client
import odc.stac

catalog = Client.open("https://earth-search.aws.element84.com/v1")
items = catalog.search(
    collections=["sentinel-2-c1-l2a"],
    bbox=(-96, 44, -90, 48),
    datetime="2025-01-01/2025-01-31",
).item_collection()

cube = odc.stac.load(
    items,
    bands=["red", "green", "blue"],
    crs="EPSG:32615", resolution=30,
    chunks={"time": 1, "x": 2048, "y": 2048},
)
small = cube.sel(
    x=slice(574000, 579000), y=slice(5199000, 5194000),
).compute()
```

<p class="sequence">retrieve items → build lazy array / task graph → read selected pixels</p>
<p class="caption">stackstac follows the same item-first pattern</p>

<!--
1:10–3:05

- Familiar pattern: search STAC → collect matching items → build an array → select pixels.
- odc-stac and stackstac established this useful array interface; pixel reads are lazy.
- `item_collection()` retrieves all matching metadata before loading; Dask can prune later work.
- In my large-scale workflow, repeated searches and large task graphs became obstacles.
- Can we defer item discovery too, when the eventual selection may be small?
-->

---
layout: center
class: bg-primary
title: What if we also defer STAC searches?
---

# We already defer reading pixels.
# What if we also defer STAC searches?


<!--
3:05–4:05

- Define the view first; query matching assets for requested spatial chunks and time steps.
- Carry query criteria instead of the complete item collection.
- Opening still needs a prepared index, representative items, COG headers, timestamps, and coordinates.
- Dask remains available for scheduling and downstream work.
- Filters, grouping, extent, and grid define the view; `.sel(...)` selects within it.
-->

---
layout: none
class: bg-black
title: Super duper
---

<script setup>
import duperOfficeGif from './assets/deck/duper-office.gif?url'
</script>

<img :src="duperOfficeGif" alt="Super duper — The IT Crowd" class="absolute inset-0 w-full h-full object-contain" />

<!--
Source: https://tenor.com/vZ9l.gif
-->

---
layout: default
class: stac-intro-slide
title: Cloud-native data needs cloud-native metadata
---

# Cloud-native data needs cloud-native metadata

<AssetImage asset="stac-geoparquet/cloud-native-data-formats.excalidraw.png" alt="Cloud-native geospatial formats: COG, COPC, Zarr, and GeoParquet" class="stac-intro-image" />

<p class="caption">STAC describes the assets. stac-geoparquet puts those descriptions in GeoParquet.</p>
<p class="footnote">Image: <a href="https://www.gadom.ski/presentations/2025-11-04-stac-geoparquet.html#/4/1">Pete Gadomski · Cloud-Native Geospatial Metadata with stac-geoparquet</a></p>

<!--
4:05–4:25

- Cloud-hosted data needs metadata: location, acquisition time, asset links.
- STAC describes the assets; stac-geoparquet stores those descriptions in GeoParquet.
- This talk uses COG pixels and GeoParquet metadata.
- Credit Pete Gadomski for the diagram.
-->

---
layout: default
class: stac-intro-slide
title: STAC items as a queryable table
---

# STAC items as a queryable table

<AssetImage asset="stac-geoparquet/stac-data-storage.excalidraw.png" alt="STAC JSON properties such as datetime, proj:code, and eo:cloud_cover become top-level columns in stac-geoparquet" class="stac-intro-image" />

<p class="caption">One row per item · geometry and properties as columns</p>
<p class="footnote">Image: <a href="https://www.gadom.ski/presentations/2025-11-04-stac-geoparquet.html#/4/2">Pete Gadomski · Cloud-Native Geospatial Metadata with stac-geoparquet</a></p>

<!--
4:25–4:45

- One row per STAC item; geometry, scene properties, and asset metadata as columns.
- Query space, time, and scene properties with DuckDB without fetching individual JSON documents.
- The table holds metadata and links; pixels stay in the COGs.
- Credit Pete Gadomski for the diagram.
-->

---
layout: default
title: GTI worked. Packaging was the obstacle.
---

# GDAL's GTI driver works. Packaging is an obstacle.

GTI = GDAL Raster Tile Index: query a tile index, then read and mosaic matching rasters.

```python
import rioxarray

red = rioxarray.open_rasterio(
    "items.parquet",
    driver="GTI",
    LOCATION_FIELD="assets.red.href",
    FILTER="datetime >= '2026-05-01' AND datetime < '2026-06-01'",
    RESX=10, RESY=10,
    chunks={"x": 2048, "y": 2048},
)
```

<p class="caption">Rasterio's PyPI wheels omit GDAL's Parquet/GeoParquet driver.<br>Build rasterio from source against a GDAL build with Parquet support.</p>

[hrodmn/stac-gti-xarray](https://github.com/hrodmn/stac-gti-xarray)

<!--
4:45–6:00

- GTI: GDAL Raster Tile Index. Query a GeoParquet index, then read, reproject, and mosaic rasters.
- `LOCATION_FIELD` points to the red asset URL; Python loops over dates and bands.
- The prototype worked; distributing the native build was the obstacle.
- Rasterio's PyPI wheels lack the Parquet driver. This workflow needs rasterio built against a Parquet-enabled GDAL.
- Installing a separate GDAL won't add drivers to the wheel's bundled GDAL.
- That packaging experience motivated a narrower, GDAL-free implementation.
-->

---
layout: center
class: wake-up-slide
title: lazycogs wakes up
---

<ClosingLogo class="wake-up-wordmark" />

<!--
6:00–6:10

- Pause for the logo to wake up, then introduce the array interface.
-->

---
layout: default
class: array-slide hls-array-slide
title: lazycogs — a year of HLS as an array
---

# lazycogs

> lazily construct an array from COGS indexed in STAC items

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
```

</div>
<div class="array-output">
<AssetFrame asset="hls/array.html" title="Actual verified 2025 HLS Xarray HTML representation" />
</div>
</div>

<p class="caption">≈15 TB logical array size · no pixel windows read yet</p>
<p class="footnote">Direct HLS S3 access requires AWS us-west-2 compute; use the <a href="https://developmentseed.org/lazycogs/notebooks/hls-https/">HTTPS notebook</a> elsewhere.</p>

<!--
6:10–8:30

- A year of Sentinel-derived HLS: 365 daily steps, three RGB bands, 30 m, UTM zone 15N.
- About 15 TB if we materialized the dense array; no pixel windows loaded yet.
- Opening queries metadata, reads COG headers, and builds coordinates.
- HLS pixel access needs NASA Earthdata credentials; these saved outputs need no live access.
- Coordinates: x ascending, y descending.
-->

---
layout: cover
class: foundations-slide
background: /images/theme/Tanezrouft_Basin.jpg
dim: true
title: Built on these tools
---

# Built on these tools

<div class="foundation-panels">
<section class="foundation-panel">
<h3>obstore</h3>
<p>Cloud object storage</p>
<p>Authenticated, asynchronous byte-range reads from the stores holding our COGs.</p>
<a href="https://github.com/developmentseed/obstore">developmentseed/obstore</a>
</section>
<section class="foundation-panel">
<h3>async-geotiff</h3>
<p>COG reads without GDAL</p>
<p>Read GeoTIFF metadata and fetch and decode the blocks needed for a pixel window, asynchronously.</p>
<a href="https://github.com/developmentseed/async-geotiff">developmentseed/async-geotiff</a>
</section>
<section class="foundation-panel">
<h3>rustac</h3>
<p>STAC metadata queries</p>
<p>Search STAC GeoParquet with DuckDB to find assets matching our space, time, and scene filters.</p>
<a href="https://github.com/stac-utils/rustac-py">stac-utils/rustac-py</a>
</section>
</div>

<!--
8:30–9:00

- obstore: authenticated cloud storage and asynchronous byte-range reads.
- async-geotiff: COG metadata and pixel reads without GDAL.
- rustac: STAC GeoParquet queries through DuckDB.
- lazycogs adds the target grid, mosaicking, and Xarray interface.
-->

---
layout: default
title: Load one point
---

# Load one point

```python
point = hls.sel(x=576465, y=5196159, method="nearest").compute()

# Rows: dates. Columns: B04 (red), B03 (green), B02 (blue).
point = point.where(point != -9999)
table = point.transpose("time", "band").to_pandas()
table.head(4)
```

| time | B04 (red) | B03 (green) | B02 (blue) |
|---|---:|---:|---:|
| 2025-01-01 | 1240 | 980 | 760 |
| 2025-01-02 | NaN | NaN | NaN |
| 2025-01-03 | 1310 | 1020 | 810 |
| 2025-01-04 | NaN | NaN | NaN |

<p class="caption">Illustrative pseudo values, not captured output · 365 dates × 3 bands<br>Stored digital numbers · missing observations become NaN</p>

<!--
9:00–10:30

- Select one point across 365 dates and three bands, without loading the full spatial array.
- Query index → find COGs → read windows/overviews → reproject and mosaic.
- A point request still fetches compressed COG tiles, not one pixel's worth of bytes.
- The table uses illustrative values; stored DNs, no cloud filter or verified reflectance scaling.
- Mask -9999 nodata to NaN before pandas conversion; keep missing dates.
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

<p class="caption">5 km window · scene cloud cover &lt;30%</p>

<!--
10:30–12:45

- Reopen with scene cloud cover <30%; `hls_args` reuses the earlier setup without its old filter.
- Select a 5 km window on April 7, July 14, and October 10.
- Scene cloud cover: 4%, 6%, 0%. Scene filtering doesn't guarantee clear pixels.
- Show seasonal differences; don't treat this as quantitative phenology.
- New filter, new array; `.sel(...)` chooses a subset within it.
- Optional: `.lazycogs.explain()` queries the index without reading pixels.
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
12:45–14:15

- Back to Southwest Sentinel-2, March–June 2025.
- Prepare our own index with `rustac.search_to`, instead of using the published HLS index.
- Save metadata locally; asset links still point to remote COG pixels.
- Prepare the index once, then open different array views and compute subsets.
- Public Sentinel-2 assets need no signed storage requests.
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
    mosaic_method=lazycogs.FirstMethod,
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
14:15–18:45

- Choose scene filters, monthly grouping, extent, target CRS, and resolution.
- Four months, three bands, 300 m: about 333 MB logical size.
- Combine source projections on one EPSG:3310 grid, using COG overviews and nearest-neighbor sampling.
- Order scenes by cloud cover, then take the first valid pixel. No monthly averaging or pixel cloud mask.
- Click: chunk to 1024 × 1024, select May, and load asynchronously.
- Return to the opening image: the actual May result, displayed at DN 0–4000.
- Change the view's definition by opening another array; use `.sel(...)` within this one.
-->

---
layout: default
class: closing-slide
title: lazycogs — discussion
---

<ClosingLogo class="closing-wordmark" />

<p class="closing-message"><strong>stay lazy</strong></p>

<div class="closing-links">
<div>
<a class="docs-link" href="https://developmentseed.org/lazycogs">developmentseed.org/lazycogs</a>
</div>
<CurrentUrlQRCode url="https://developmentseed.org/lazycogs" :width="160" :height="160" />
</div>

<!--
18:45–20:00 · then 5 minutes for discussion

- Scope: compatible STAC-indexed COGs, without aiming for GDAL feature parity.
- Index layout and chunk sizes still matter.
- Try it on your archives; report unsupported workflows, unexpected results, and rough edges.
- Docs: https://developmentseed.org/lazycogs/
- Issues: https://github.com/developmentseed/lazycogs/issues
- What archive and array view do you want to work with?
-->
