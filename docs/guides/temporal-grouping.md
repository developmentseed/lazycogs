# Temporal grouping

By default, each time step in a lazycogs DataArray represents one calendar day. When multiple scenes were acquired on the same day, the [mosaic method](mosaic-methods.md) merges them into a single value per pixel.

The `time_period` parameter lets you aggregate across longer windows. Setting `time_period="P1W"` means each time step represents one ISO calendar week; all scenes within that week are mosaicked together into a single composite. This is more efficient than building a daily array and then reducing along the time axis, because lazycogs only reads the pixels it actually needs for each composite — it never materializes the individual daily values before reducing them.

## Supported values

| `time_period` | Window | Time coordinate |
|---|---|---|
| `"P1D"` (default) | Calendar day | `YYYY-MM-DD` |
| `"P1W"` | ISO calendar week (Monday through Sunday) | Monday of the week |
| `"P1M"` | Calendar month | First day of the month |
| `"P1Y"` | Calendar year | January 1st |
| `"P16D"` | 16-day fixed window | Aligned to 2000-01-01 epoch |
| `"PnD"` (any n) | n-day fixed window | Aligned to 2000-01-01 epoch |

The `P16D` period is useful for Landsat (16-day orbital repeat) and MODIS-style composites. Any integer multiple of days is supported via `PnD`.

## Why this is more efficient than post-hoc resampling

Suppose you want monthly composites over a year. You could:

1. Open a daily DataArray (`time_period="P1D"`, 365 steps) and call `.resample(time="MS").max()`
2. Open directly as monthly (`time_period="P1M"`, 12 steps)

Option 1 materializes every daily value before the reduction — even for days with no data. Option 2 queries only the items that fall within each month when a pixel is read. For large archives with sparse coverage, option 2 can be orders of magnitude faster.

## Common patterns

**Weekly low-cloud composite:**

```python
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    time_period="P1W",
    sortby=["eo:cloud_cover"],       # sort by cloud cover ascending (scene-level)
)
# Each time step is a composite that preferentially draws from the least cloudy
# scenes acquired during that ISO week. Scene-level cloud cover sorting reduces
# overall cloud contamination but does not remove individual cloudy pixels unless
# they are masked to nodata in the source data.
```

**Monthly composite:**

```python
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    time_period="P1M",
)
```

**16-day Landsat-style composite:**

```python
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=30.0,
    time_period="P16D",
)
```

## Combining with mosaic methods

`time_period` and `mosaic_method` work together. A common idiom is weekly windows with `FirstMethod` (default) sorted by ascending cloud cover — this draws each pixel from the least cloudy scene available in the week and stops reading further scenes once all output pixels are filled. Because `eo:cloud_cover` is a scene-level metric, this reduces overall cloud contamination but does not remove individual cloudy pixels:

```python
from lazycogs import FirstMethod

da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    time_period="P1W",
    sortby=["eo:cloud_cover"],
    mosaic_method=FirstMethod,
)
```

See also: [Demo notebook](../demo.ipynb) for a worked example of monthly compositing, [Mosaic methods guide](mosaic-methods.md), [API reference for open()](../api/open.md)
