"""lazycogs: lazy xarray DataArrays from STAC COG collections."""

from lazycogs._core import open, open_async  # noqa: A004
from lazycogs._executor import set_reproject_workers
from lazycogs._explain import (  # noqa: F401 — registers da.stac_cog accessor
    ChunkRead,
    CogRead,
    ExplainPlan,
    StacCogAccessor,
)
from lazycogs._grid import align_bbox
from lazycogs._mosaic_methods import (
    CountMethod,
    FirstMethod,
    HighestMethod,
    LowestMethod,
    MeanMethod,
    MedianMethod,
    MosaicMethodBase,
    StdevMethod,
)
from lazycogs._store import store_for

__all__ = [
    "ChunkRead",
    "CogRead",
    "CountMethod",
    "ExplainPlan",
    "FirstMethod",
    "HighestMethod",
    "LowestMethod",
    "MeanMethod",
    "MedianMethod",
    "MosaicMethodBase",
    "StdevMethod",
    "align_bbox",
    "open",
    "open_async",
    "set_reproject_workers",
    "store_for",
]
