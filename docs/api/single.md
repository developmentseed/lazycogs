# Single COG / item

Read a single Cloud-Optimized GeoTIFF or the assets of one STAC item at their
native grid — no reprojection or mosaicking. Use `lazycogs.open` for a
reprojected mosaic across a whole collection.

Opening and selecting read headers but no pixels. Select before loading to
limit the spatial window and, for an item, the assets read:

```python
da = lazycogs.open_item(item, bands=["red", "nir"])
window = da.sel(band="nir").isel(y=slice(100, 356), x=slice(200, 456))
pixels = window.load()
```

Reads use raw values at full native resolution. For a multiband COG, the
spatial window is read for all bands before applying the band selection.

!!! tip "See also"
    [Loading single item on native grid](../notebooks/single-item.ipynb)

::: lazycogs.open_cog

::: lazycogs.open_item

::: lazycogs.open_cog_async

::: lazycogs.open_item_async

