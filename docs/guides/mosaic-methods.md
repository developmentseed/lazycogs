# Mosaic methods

When multiple COG tiles overlap within a single time step, lazycogs needs a rule for deciding which pixel value to use. That rule is a **mosaic method**.

Mosaic methods apply within a time step, not across time steps. If you have ten Sentinel-2 scenes acquired on the same day and they all overlap your area, the mosaic method decides how to merge them into one value per pixel. Pixels that fall outside all scenes remain nodata regardless of which method you choose.

!!! note
    Mosaic methods handle nodata and overlap, not cloud masking. If a pixel is flagged as cloud in the source COG but is not set to nodata, the mosaic method will treat it as valid data. Cloud masking must be applied upstream — typically by filtering items with a cloud cover threshold before passing the parquet file to `lazycogs.open()`.

## Choosing a method

| Method | Best for | Early exit (stops when pixels filled) |
|---|---|---|
| `FirstMethod` (default) | Low-cloud priority composites, newest-first sort | Yes |
| `HighestMethod` | Peak reflectance, maximum value composites | No |
| `LowestMethod` | Minimum cloud shadow, lowest temperature | No |
| `MeanMethod` | Smooth temporal mean | No |
| `MedianMethod` | Outlier-robust mean | No |
| `StdevMethod` | Variability / change detection | No |
| `CountMethod` | Data coverage, observation density | No |

### `FirstMethod`

The default. For each output pixel, reads scenes in order and uses the value from the first scene that has valid (non-nodata) data at that location. Stops reading further scenes once every pixel in the chunk is filled. This short-circuit behavior makes it the most efficient choice when tiles have low overlap or when you sort by cloud cover before opening.

Typical usage: weekly or monthly low-cloud composites, sorted by `eo:cloud_cover` ascending. Because `eo:cloud_cover` is a scene-level metric, this preferentially reads from the least cloudy scenes first — but individual cloudy pixels from those scenes will still appear in the output unless the source COGs already have those pixels set to nodata.

### `HighestMethod` / `LowestMethod`

Reads all overlapping scenes and takes the maximum or minimum value per pixel. Useful for peak reflectance composites or for suppressing cloud shadows (minimum NIR or visible reflectance over a period where clouds are bright).

### `MeanMethod`

Averages all overlapping valid pixels. Appropriate for producing smooth composites where outliers have already been removed by a cloud filter.

### `MedianMethod`

Takes the median of all overlapping valid values. More robust than mean when a small number of corrupted or cloudy pixels slip through. Because it must accumulate all tile values before computing the median, it holds all overlapping tiles in memory simultaneously — more memory-intensive than methods that stream one tile at a time.

### `StdevMethod`

Computes the per-pixel standard deviation across all overlapping tiles. Useful for change detection or for identifying areas of high temporal variability. Like `MedianMethod`, it accumulates all tiles before computing, so memory use scales with the number of overlapping scenes.

### `CountMethod`

Returns the number of valid (non-nodata) observations that contributed to each pixel. Useful for generating data coverage masks or quality flags alongside a primary composite.

## Specifying a method

Pass the method class (not an instance) to `open()`:

```python
import lazycogs
from lazycogs import HighestMethod

da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    mosaic_method=HighestMethod,
)
```

## Performance note

`FirstMethod` is the most efficient option for priority-mosaic workflows because it short-circuits as soon as all output pixels are filled. `MedianMethod` and `StdevMethod` must hold all overlapping tiles in memory at once — if your area has high scene overlap (common at high latitudes), these methods will use significantly more memory per chunk.

See also: [API reference for mosaic methods](../api/mosaic.md)
