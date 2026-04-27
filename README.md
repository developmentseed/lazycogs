![lazycogs](./lazycogs.svg)

Open a lazy `(time, band, y, x)` xarray DataArray from thousands of cloud-optimized geotiffs (COGs). No GDAL required.

## What is lazycogs?

[stackstac](https://stackstac.readthedocs.io) and [odc-stac](https://odc-stac.readthedocs.io) established the pattern that lazycogs builds on: take a STAC item collection and expose it as a spatially-aligned xarray DataArray ready for dask-parallel computation. Both are excellent tools that cover most satellite imagery workflows well. They rely on the trusty combination of rasterio and GDAL for data i/o and warping operations.

lazycogs takes the same approach but replaces GDAL and rasterio with a Rust-native stack: [rustac](https://stac-utils.github.io/rustac-py) for STAC queries over stac-geoparquet files, [async-geotiff](https://developmentseed/async-geotiff) for COG i/o, and [obstore](https://developmentseed.org/obstore) for cloud storage access.

The result is a tool that can instantly expose a lazy xarray DataArray view of massive STAC item archives in any CRS and resolution. Each array operation triggers a targeted spatial query on the stac-geoparquet file to find only the assets needed for that specific chunk — no upfront scan of every item required.

One constraint worth naming: lazycogs only reads Cloud Optimized GeoTIFFs. If your assets are in another format, this is not the right tool.

| Task | Library |
|---|---|
| STAC search + spatial indexing | `rustac` (DuckDB + geoparquet) |
| COG I/O | `async-geotiff` (Rust, no GDAL) |
| Cloud storage | `obstore` |
| Reprojection | `pyproj` + numpy |
| Lazy dataset construction | xarray `BackendEntrypoint` + `LazilyIndexedArray` |

## Installation

Not yet published to PyPI. Install directly from GitHub:

```bash
pip install lazycogs
```

## Example

```python
import rustac
import lazycogs
from pyproj import Transformer

# set a target CRS and extent
dst_crs = "EPSG:32615"
dst_bbox = (380000.0, 4928000.0, 420000.0, 4984000.0)

# transform to 4326 for STAC search
transformer = Transformer.from_crs(dst_crs, "epsg:4326", alwaysxy=True)
bbox_4326 = transformer.transform_bounds(*dst_bbox)

# Search a STAC API and cache results to a local stac-geoparquet file.
await rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=bbox_4326,
)

# Open a fully lazy (time, band, y, x) DataArray. No COGs are read yet.
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
)
```

## Documentation

- [Home](https://developmentseed.github.io/lazycogs/) — quickstart and full usage guide
- [Example: Midwest US daily Sentinel-2 array](https://developmentseed.org/lazycogs/notebooks/demo_midwest_daily/)
- [Example: Southwest US monthly low-cloud Sentinel-2 array](https://developmentseed.org/lazycogs/notebooks/demo_southwest_monthly/)
- [Architecture](https://developmentseed.org/lazycogs/architecture/)
- [Contributing](CONTRIBUTING.md)
