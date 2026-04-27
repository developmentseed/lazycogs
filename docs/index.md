![lazycogs](./lazycogs.svg)

Open a lazy `(time, band, y, x)` xarray DataArray from thousands of cloud-optimized GeoTIFFs. No GDAL required.

--8<-- "docs/includes/dataarray_repr.html"

## What is lazycogs?

lazycogs lets you materialize a lazy xarray DataArray view of massive STAC-indexed data archives in any CRS and resolution. Opening the array is nearly instant because no COGs are read until you request pixels. lazycogs queries the stac-geoparquet dataset using [rustac](https://stac-utils.github.io/rustac-py) to find only the COGs that intersect a spatial and temporal selection, fetches only the relevant pixel windows using [async-geotiff](https://developmentseed.org/async-geotiff), and reprojects into your target grid.

**Note:** lazycogs only reads GeoTIFFs. If your assets are in another format, lazycogs is not the right tool.

Here is a summary of the libraries lazycogs uses for each step:

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
pip install git+https://github.com/hrodmn/lazycogs.git
```

## Minimal example

```python
import rustac
import lazycogs
from pyproj import Transformer

dst_crs = "EPSG:5070"
dst_bbox = (-400_000, 2_500_000, -200_000, 2_700_000)

transformer = Transformer.from_crs(dst_crs, "epsg:4326", always_xy=True)
bbox_4326 = transformer.transform_bounds(*dst_bbox)

# Search a STAC API and cache results to a local stac-geoparquet file.
await rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-c1-l2a"],
    datetime="2025-06-01/2025-08-31",
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

Get started with the [Quickstart](quickstart.ipynb). Evaluating lazycogs against alternatives? See [Performance](performance.md).
