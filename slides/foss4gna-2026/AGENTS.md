# FOSS4G North America 2026 presentation

This directory contains the talk-specific Slidev deck. Shared theme, dependencies, and assets live in the parent `slides/` directory.

## Working guidance

- Use the submitted proposal below as the reference for the talk's scope and promises. Preserve the submission verbatim; distinguish it from revised slide copy.
- Develop the talk structure in `outline.md` and the deck in `slides.md`.
- Use ~/workspace/devseed/collection-discovery-foss4gna-2026/ for inspiration on slidev features/capabilities like animations, transitions, etc.
- Draw examples from `../../docs/notebooks/`. Check notebook claims against the current `../../README.md`, `../../ARCHITECTURE.md`, and implementation before using them in slides.
- Lead with the speaker's vision: “Query satellite imagery collections like everyone wants to: as a big array”.
- Preserve the inspiration: in a past large-scale satellite imagery analysis project, the speaker wanted to quickly define a massive array from STAC metadata and distribute work chunkwise. stackstac and odc-stac made this theoretically possible, but the speaker encountered massive Dask task graphs that did not scale as desired, and workflows that usually involved many STAC API queries. These are observations about that project's workflows, not universal claims about those libraries. STAC GeoParquet suggested a way to distribute the metadata/query work once rustac enabled API-like searches over the files. Distinguish this original motivation from the later GTI packaging experience. Do not claim that lazycogs eliminates task graphs or guarantees arbitrary distributed-worker deployment.
- Include the speaker's working GTI experiment, https://github.com/hrodmn/stac-gti-xarray (local checkout: `/home/henry/workspace/hrodmn/stac-gti-xarray`). Opening `GTI:/path/to/stac.parquet` through rioxarray worked, and the SQL `FILTER` open option supported item filtering. The speaker built GDAL from source with Parquet/GeoParquet support, then built rasterio against that installation. Distributing that native build chain and runtime setup was the blocker, motivating a tightly scoped, GDAL-free architecture. The current prototype Dockerfile instead starts from a GDAL image and builds rasterio from source. Do not imply that GTI failed at raster work or generalize this history into an unverified claim about current packaging.
- Explain why Development Seed built lazycogs, demonstrate an analysis workflow, and discuss the tradeoffs of composing narrowly scoped tools.
- Credit stackstac and odc-stac as the foundation for this pattern. Avoid blanket performance claims or presenting lazycogs as a general replacement for GDAL.
- Distinguish index preparation, array opening, and pixel reads. Opening inspects a representative item, opens a COG per selected band for metadata, and queries timestamps. It does not read pixel windows. Do not turn “lazy” into a claim of zero startup I/O or zero metadata scanning.
- Treat timings in the comparison notebook as a single-run observation, not a general benchmark. Revalidate measurements before presenting them.
- The slot is 25 minutes. Target 12 main slides (acceptable range: 10–12), with a 20-minute presentation and 5 minutes for discussion. Spend about 5 minutes on background and inspiration.
- Keep visible text minimal. Favor short code snippets, imagery, and diagrams; put narrative detail and caveats in Slidev speaker notes. The speaker will supply the narrative aloud.
- Center the main examples on a full year of HLS granules over an entire UTM zone (`hls-s3.ipynb` / `hls-https.ipynb`) and Southwest Sentinel-2 across multiple source projections (`demo_southwest_monthly.ipynb`). HLS demonstrates scale and small selections; Southwest demonstrates alignment across source CRSs. Keep the separate HLS NDVI analysis and detailed comparisons in backup.
- Show each example's `lazycogs.open(...)` call and the actual Xarray HTML DataArray repr. The speaker prefers HTML reprs for arrays; do not replace them with plain-text reprs or hand-built summaries. Hide auth, grid inspection, and other setup behind preconfigured variables and link the full notebooks. Verify the year filter, collection, logical size, and source CRS distribution before making quantitative claims. The S3 HLS notebook explicitly selects 2025; reconcile the HTTPS notebook's missing year predicate/client argument before using it as that reference.
- Add a visible footnote to the HLS S3 example: direct HLS S3 access requires compute in AWS us-west-2; use the HTTPS notebook elsewhere. Keep authentication setup out of the main snippet.
- Use the combined relaxed-squares and “lazycogs” SVG (`../../logo-text.svg`) on the title slide instead of a separate logo and printed heading. Subtitle: “super lazy analysis of massive COG archives”, matching the remainder of the submitted session title. `../../logo.svg` is the icon alone; `../../lazycogs.svg` is the text wordmark alone.
- Match the title-slide detail panel to `~/workspace/devseed/collection-discovery-foss4gna-2026/slides.md`: FOSS4GNA 2026, Sacramento, CA, 2026-11-03, but list only Henry Rodman as presenter. Preserve its panel typography and layout.
- Audience prerequisites are not yet specified.

## Submitted presentation materials

### Proposal Title

lazycogs: super lazy analysis of massive COG archives

### Proposal Abstract

Maximum 40 words; used in marketing materials to promote the talk.

lazycogs is a lightweight tool for representing archives of COGs as lazy Xarray DataArrays in Python. It’s open source, using Rust and leveraging STAC GeoParquet, created by Development Seed. Come learn a fresh approach for STAC-indexed COGs!

### Proposal Description

Maximum 800 words; used to determine acceptance or rejection.

lazycogs is a Python package built by Development Seed for analyzing massive collections of cloud-optimized geotiffs. It builds on the pattern established by stackstac and odc-stac: take a STAC item collection and expose it as a spatially-aligned Xarray DataArray ready for parallel computation. stackstac and odc-stac are excellent tools that rely on the trusty combination of rasterio and GDAL for data i/o and warping operations and they cover many satellite imagery workflows well, but when you push them to work with thousands of assets they can struggle.

lazycogs takes the same approach but replaces GDAL and rasterio with python wrapped Rust libraries: rustac for STAC queries over stac-geoparquet files, async-geotiff for COG i/o, and obstore as the default cloud storage integration. The result is a tool that can instantly expose an Xarray DataArray view of massive STAC item archives in any CRS and resolution where the bytes are only fetched when you operate on the data for a given spatial/temporal window (i.e. lazy). Each array operation triggers a targeted spatial query on the stac-geoparquet file to find only the assets needed for that specific chunk - no upfront scan of every item required.

In this talk you will learn about why we built lazycogs, what it can be used to do, and how the pattern of using narrowly scoped tools can be a useful alternative to systems built on monolithic projects.
