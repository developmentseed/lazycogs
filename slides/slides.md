---
title: lazycogs
info: A lazy xarray view of cloud-optimized GeoTIFF collections
class: text-center
highlighter: shiki
drawings:
  persist: false
theme: ./theme
layout: cover
background: '#F5F2ED'
---

# lazycogs

::subtitle::
Lazy, cloud-native raster mosaics

<LogoHorPos position="bottom-right" height="28px" />

---

# What lazycogs does

lazycogs opens a lazy `(band, time, y, x)` xarray DataArray from collections of Cloud Optimized GeoTIFFs.

- Uses STAC metadata to locate the assets needed for a requested spatial chunk
- Reads COG data without GDAL, using a Rust-native I/O stack
- Lets xarray and Dask drive computation when array chunks are requested

---

# A lazy read path

1. Search a STAC API and save results to a stac-geoparquet file
2. Open that file with a target extent, CRS, and resolution
3. Select or compute array chunks as needed

The presentation deck is a starting point. Add talk-specific material here.

---

# Future talk content

Placeholder for a future presentation.
