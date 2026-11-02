---
title: lazycogs
info: A lazy xarray view of cloud-optimized GeoTIFF collections
highlighter: shiki
drawings:
  persist: false
theme: ../theme
layout: title
image: /images/theme/lena-delta.jpg
---

# lazycogs

::subtitle::
Lazy, cloud-native raster mosaics

<DecorativeRectangle
  width="50%"
  height="40%"
  zIndex=20
  :position="{
    bottom: '2%',
    right: '2%',
  }"
  :customStyle="{ mixBlendMode: 'multiply' }"
>
  <div w-full h-full relative flex flex-col items-end justify-end p-4 text-white text-right>
    <p text-3xl font-300>
      FOSS4G North America 2026
    </p>
  </div>
</DecorativeRectangle>

<LogoHorPos position="top-left" height="24px" />

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
