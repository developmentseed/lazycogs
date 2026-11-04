# Verified HLS open snippets

Both calls were executed against the same current 2025 HLSS30_2.0 GeoParquet partitions. Setup variables hide auth, archive preparation, and grid alignment. The monthly open is a separate array definition.

```python
baseline = lazycogs.open(
    href=index_href, bbox=zone15_bounds_aligned, crs='EPSG:32615',
    resolution=30, bands=['B04', 'B03', 'B02'], time_period='P1D',
    datetime='2025-01-01T00:00:00Z/2025-12-31T23:59:59Z',
    filter=zone15_filter, duckdb_client=index_client, store=hls_store,
)

monthly = lazycogs.open(
    href=index_href, bbox=zone15_bounds_aligned, crs='EPSG:32615',
    resolution=30, bands=['B04', 'B03', 'B02'], time_period='P1M',
    datetime='2025-01-01T00:00:00Z/2025-12-31T23:59:59Z',
    filter="proj:epsg = '32615' AND eo:cloud_cover < 50",
    sortby='eo:cloud_cover', duckdb_client=index_client, store=hls_store,
)
```

Verified baseline dimensions: {'band': 3, 'time': 365, 'y': 310967, 'x': 22266}. Monthly time coordinates: 12. The filter predicate was run against the current index and sample items expose `eo:cloud_cover`.
