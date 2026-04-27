# STAC item queries

`lazycogs.open()` queries a stac-geoparquet file to determine what goes into the output DataArray: which time steps exist, which scenes fill each pixel, and what bands are available. The query parameters control all of this — they are not just filters applied after the fact, they define the shape of the array.

## Caching a STAC API search to geoparquet

lazycogs reads from a local (or cloud-hosted) stac-geoparquet file — it never calls a STAC API directly. Use `rustac.search_to()` to run the search once and write the results to a file:

```python
from pathlib import Path
import rustac

PARQUET = "data/items.parquet"

if not Path(PARQUET).exists():
    Path(PARQUET).parent.mkdir(exist_ok=True)
    await rustac.search_to(
        PARQUET,
        href="https://earth-search.aws.element84.com/v1",
        collections=["sentinel-2-c1-l2a"],
        datetime="2025-06-01/2025-09-30",
        bbox=[-95.0, 38.0, -88.0, 45.0],
        limit=100,
    )
```

The `if not PARQUET.exists()` guard means re-running the notebook or script skips the API call. After that, all lazycogs queries read directly from the parquet file. `rustac.search_to()` also accepts `filter=` and `sortby=` if you want to push those filters down to the STAC API at search time rather than applying them later in lazycogs.


## How queries shape the output

When you call `lazycogs.open()`, it runs two DuckDB queries against the parquet file before returning:

1. Band discovery: finds the asset keys present in the first matching item (skipped if you pass `bands=`).
2. Time-step discovery: scans all matching items to find the unique time steps (grouped by `time_period`).

Every subsequent pixel read re-runs a narrower version of the same query — filtered to a single time step — so the same parameters that shape the array also govern which scenes are read at access time.

## Spatial filter

`bbox` is required. It is the spatial extent of the output array in the target `crs`. lazycogs reprojects it to EPSG:4326 internally before querying the parquet, so items are filtered by their footprint against the bounding box.

## Temporal filter

`datetime` accepts an RFC 3339 datetime or date range:

```python
# Single point in time
da = lazycogs.open("items.parquet", ..., datetime="2023-06-15")

# Range
da = lazycogs.open("items.parquet", ..., datetime="2023-01-01/2023-12-31")
```

`datetime` is a pre-filter applied before time-step discovery. Only items whose acquisition time falls within the range are considered; the output time axis is built from those items alone. Narrowing `datetime` reduces both the number of DuckDB rows scanned and the number of time steps in the output array.

See the [temporal grouping guide](temporal-grouping.md) for how items are bucketed into time steps once the `datetime` filter is applied.

## CQL2 filters

`filter` accepts a [CQL2](https://docs.ogc.org/is/21-065r2/21-065r2.html) expression that is forwarded to every DuckDB query. Use it to filter by any STAC item property stored in the parquet:

```python
# Items with less than 20% cloud cover (EO extension property)
da = lazycogs.open("items.parquet", ..., filter="eo:cloud_cover < 20")

# Items from a specific platform
da = lazycogs.open("items.parquet", ..., filter="platform = 'sentinel-2b'")

# Combine conditions
da = lazycogs.open(
    "items.parquet",
    ...,
    filter="eo:cloud_cover < 15 AND platform = 'sentinel-2a'",
)
```

CQL2-JSON dict is also accepted if you are building filters programmatically:

```python
da = lazycogs.open(
    "items.parquet",
    ...,
    filter={"op": "lt", "args": [{"property": "eo:cloud_cover"}, 20]},
)
```

The filter applies at both the time-step discovery stage and at each pixel read. Time steps that have no items passing the filter after the spatial/temporal pre-filter are excluded from the output array.

**Note:** CQL2 filters operate on item-level properties, not pixel values. `eo:cloud_cover < 20` means the scene-level cloud cover estimate is below 20%, not that individual cloudy pixels are masked. To skip cloudy pixels, use a `nodata`-masked source dataset or apply a pixel-level mask after loading.

## Selecting items by ID

`ids` restricts the query to a specific list of STAC item IDs:

```python
da = lazycogs.open(
    "items.parquet",
    ...,
    ids=["S2A_MSIL2A_20230601", "S2B_MSIL2A_20230604"],
)
```

This is useful when you already know which items you want — for example, from a prior STAC API search — and want to skip any further filtering.

## Selecting bands

By default, lazycogs discovers bands automatically from the first matching item: it prefers assets whose media type contains `"image/tiff"` or whose roles include `"data"`, and falls back to all assets. Pass `bands=` to override this:

```python
# Only load the red, green, and blue bands
da = lazycogs.open("items.parquet", ..., bands=["B04", "B03", "B02"])
```

The order of `bands` controls the order of the `band` coordinate in the output array.

## Sorting and mosaic order

`sortby` controls the order in which scenes are presented to the mosaic method within each time step. It accepts a field name string, a list of field names (with optional `+`/`-` prefix), or a list of `{"field": ..., "direction": ...}` dicts:

```python
# Ascending cloud cover — least cloudy scenes read first
da = lazycogs.open("items.parquet", ..., sortby=["eo:cloud_cover"])

# Descending — most recent scene wins (useful for latest-pixel composites)
da = lazycogs.open("items.parquet", ..., sortby=["-datetime"])

# Dict form
da = lazycogs.open(
    "items.parquet",
    ...,
    sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
)
```

The default mosaic method (`FirstMethod`) stops reading once all output pixels are filled, so the sort order directly determines which scene contributes each pixel. Sorting by ascending `eo:cloud_cover` means the least cloudy scene fills each pixel first. See the [mosaic methods guide](mosaic-methods.md) for how different methods use the scene order.

## Hive-partitioned parquet

By default, `lazycogs.open()` expects `href` to point to a single `.parquet` or `.geoparquet` file and creates a plain `DuckdbClient()` internally. If your STAC items are stored as a hive-partitioned directory (e.g. `year=2023/month=01/...`), pass a pre-configured client with `use_hive_partitioning=True`:

```python
from rustac import DuckdbClient
import lazycogs

client = DuckdbClient(use_hive_partitioning=True)

da = lazycogs.open(
    "s3://bucket/stac/",            # directory, not a single file
    duckdb_client=client,
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
```

DuckDB skips partition directories that cannot match the spatial and temporal filters, which can dramatically reduce the number of parquet files scanned on large archives. All other query parameters (`datetime`, `filter`, `sortby`, etc.) work the same way.

See also: [API reference for open()](../api/open.md), [Temporal grouping guide](temporal-grouping.md), [Mosaic methods guide](mosaic-methods.md)
