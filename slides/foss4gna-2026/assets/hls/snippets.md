# HLS open snippets and verification status

## Baseline: executed, saved 2025 S3 notebook output

Source: `docs/notebooks/hls-s3.ipynb`, cell 15 (actual saved Xarray output exported as `array.html`). Its preceding call explicitly selects `HLSS30_2.0`, 2025-01-01 through 2025-12-31, `proj:epsg=32615`, 30 m, daily timestamps, and B04/B03/B02. The caller-supplied hive-aware DuckDB client is passed through. This was a historical executed notebook result, not a fresh rerun in this task.

```python
stac_search_args = {
    "href": HLS_STAC_GEOPARQUET_HREF.format(collection="HLSS30_2.0"),
    "datetime": "2025-01-01T00:00:00Z/2025-12-31T23:59:59Z",
    "filter": {"op": "=", "args": [{"property": "proj:epsg"}, epsg_code]},
    "duckdb_client": duckdb_client,
}

imagery = lazycogs.open(
    crs=dst_crs,
    bbox=dst_bbox_aligned,
    resolution=30,
    time_period="P1D",
    bands=["B04", "B03", "B02"],
    dtype=geotiff.dtype,
    nodata=geotiff.nodata,
    store=store,
    path_from_href=strip_bucket,
    **stac_search_args,
)
```

## Monthly scene-cloud filter: BLOCKED, not claimed verified

The saved sample STAC feature in `hls-s3.ipynb` has `eo:cloud_cover` (example value `99.0`), and current lazycogs forwards `filter=` to rustac. But the current archive query cannot be reproduced here: unsigned access to the public MAAP parquet object returned HTTP 403. Therefore field support/current predicate semantics and this alternate open have not been verified against the requested index. Do not use this as a verified slide snippet yet.

Candidate to verify after the parquet object is accessible:

```python
monthly = lazycogs.open(
    **common_open_args,
    time_period="P1M",
    filter="eo:cloud_cover < 50",
)
```

This would define a new array open; it does not mutate the daily array. Scene-level cloud filtering is not a pixel cloud mask.
