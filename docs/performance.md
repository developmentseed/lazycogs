# Performance

## How lazycogs differs from stackstac and odc-stac

stackstac and odc-stac both require loading STAC item metadata into memory before building the array view — a Python list of `pystac.Item` objects that must be fetched from a STAC API or read from a file upfront. lazycogs instead reads from a STAC geoparquet file on demand, querying only the rows that intersect the requested spatial and temporal extent at read time. This means array initialization is nearly instant and does not scale with archive size.

lazycogs also replaces rasterio/GDAL with [async-geotiff](https://developmentseed.github.io/async-geotiff/) and [obstore](https://developmentseed.org/obstore/latest/) for pixel I/O. Reads are fully async, and lazycogs only fetches the pixel windows it actually needs — not full scenes. When using the default `FirstMethod`, it stops reading further COGs as soon as every output pixel is filled.

## Benchmark results

Benchmarks run against `odc-stac==0.5.2` on a laptop against the Sentinel-2 C1 L2A collection over a large Midwest area (summer 2025). Results will vary across machines and network conditions.

**A note on chunking:** each library is configured with the chunk strategy that suits its architecture. lazycogs uses `chunks={"time": 1}` with no spatial chunking, because it handles spatial windowing internally. odc-stac uses `chunks={"time": 1, "x": 512, "y": 512}` because without spatial chunks, computing any spatial subset would require loading entire scenes. These configurations are not identical, but they represent the practical defaults a user would choose for each library.

| Operation | lazycogs | odc-stac |
|---|---|---|
| Load STAC items into memory | n/a | 12.22s |
| Initialize array | 0.90s | 52.05s |
| Extract point values (30 days) | 5.01s | 16.21s |
| Load spatial subset array (3 time steps) | 16.37s | 28.93s |

For full reproducible benchmarks, see the [lazycogs vs odc-stac notebook](notebooks/lazycogs-odc-stac.ipynb).
