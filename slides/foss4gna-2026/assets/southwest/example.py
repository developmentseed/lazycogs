# ruff: noqa: INP001
"""Southwest monthly array definition shown with the final assets."""

from obstore.store import HTTPStore

import lazycogs

index_path = ".cache/southwest-images/southwest-2025-full.parquet"
store = HTTPStore("https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com")

southwest_monthly = lazycogs.open(
    index_path,
    bbox=(-444000, -609000, 681000, 500000),
    crs="EPSG:3310",
    resolution=300,
    time_period="P1M",
    bands=["red", "green", "blue"],
    dtype="uint16",
    filter="eo:cloud_cover < 50",
    sortby="eo:cloud_cover",
    store=store,
)
