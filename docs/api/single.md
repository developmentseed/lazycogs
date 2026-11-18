# Single COG / item

Read a single Cloud-Optimized GeoTIFF or the assets of one STAC item at their
native grid — no reprojection or mosaicking. Use `lazycogs.open` for a
reprojected mosaic across a whole collection.

Open fetches metadata for the requested assets without decoding pixels.
Metadata read-ahead can include compressed tile bytes. `sel` and `isel` on the
returned lazy array issue no storage requests. Select before loading to limit
the spatial window and, for an item, the assets read:

```python
da = lazycogs.open_item(item, bands=["red", "nir"])
window = da.sel(band="nir").isel(y=slice(100, 356), x=slice(200, 456))
pixels = window.load()
```

Each backend read requests a bounding window of raw, full-resolution pixels.
Sparse or strided selections include gaps in that window, and TIFF tile reads
can extend beyond it. For a multiband COG, the reader fetches all bands in the
window before applying the band selection. `open_item` requires single-band
assets with identical grids and a common dtype that preserves raw values.
Use a `file://` URI to open a local COG without passing a store.

!!! tip "See also"
    [Loading single item on native grid](../notebooks/single-item.ipynb)

::: lazycogs.open_cog

::: lazycogs.open_item

::: lazycogs.open_cog_async

::: lazycogs.open_item_async

