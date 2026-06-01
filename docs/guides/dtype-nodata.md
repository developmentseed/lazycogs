# Dtype and nodata handling

`lazycogs.open()` returns a plain numeric `xarray.DataArray`. It does not return a masked array.

That means lazycogs has to make two output-level decisions:

- `dtype`: the numeric type of the returned array
- `nodata`: one scalar sentinel value for missing pixels, when that sentinel is knowable

Internal masking still happens while reading and mosaicking COGs, but the public contract is always the numeric array you get back.


## Default behavior

When you omit `dtype=` and `nodata=`, lazycogs inspects the first matching STAC item and opens one representative COG per requested band.

From that sample it resolves:

- one safe output `dtype`
- one output `nodata` value, if the sampled bands agree on one

### Dtype inference

The sampled band dtypes are promoted to one safe output dtype.

Examples:

- all `uint16` -> `uint16`
- `uint8` + `int16` -> `int16`
- any `float32` present -> `float32`
- any `float64` present -> `float64`

If lazycogs cannot promote the sampled dtypes safely, `open()` raises and asks you to pass `dtype=` explicitly.

### Nodata inference

When you omit `nodata=`:

- if sampled bands all agree on one scalar nodata value, lazycogs uses it
- if sampled bands all have `nodata=None`, lazycogs leaves output nodata unknown
- if sampled bands disagree, `open()` raises and asks you to pass `nodata=` explicitly

When nodata is known, the returned `DataArray` advertises it with:

- `da.attrs["_FillValue"]`

lazycogs intentionally does not duplicate `_FillValue` into `da.encoding` because
that collides with xarray's CF encoding step during rioxarray exports.

## What output nodata means

When lazycogs knows one scalar nodata value, that value applies to:

- uncovered pixels
- out-of-bounds pixels introduced by reprojection
- pixels where every contributing source pixel was nodata

When nodata is unknown, lazycogs does **not** attach `_FillValue` metadata. In that mode, `0` may still appear in uncovered regions as an implementation fill value, but `0` is not declared as semantic nodata.

If you need a stable sentinel for downstream analysis or export, pass `nodata=` explicitly.

## Mosaic-method interaction

`MeanMethod`, `MedianMethod`, and `StdevMethod` require floating-point output.

If you omit `dtype=`, lazycogs will auto-promote an inferred integer dtype to `float32` for those methods. If you pass an explicit integer `dtype=`, `open()` raises.

```python
from lazycogs import MeanMethod

da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    mosaic_method=MeanMethod,
)
```

## Explicit overrides win

Caller-provided `dtype=` and `nodata=` are authoritative.

```python
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    dtype="float32",
    nodata=-9999,
)
```

Use explicit overrides when you need a fixed contract across heterogeneous assets.

## Runtime validation

The startup inspection samples only one representative item. Later chunk reads still validate what they encounter.

If a later asset conflicts with the inferred output contract, compute raises instead of silently truncating values or remapping nodata.

## Quick checks

```python
da.dtype
da.attrs.get("_FillValue")
```

Those two values tell you most of what lazycogs has promised about the returned array.
