"""Shared STAC asset helpers."""

from __future__ import annotations

from typing import Any


def _preferred_data_assets(assets: dict[str, Any]) -> list[str]:
    """Return asset keys that look like raster data, or every key as fallback."""
    data_keys = [
        key
        for key, asset in assets.items()
        if "data" in asset.get("roles", []) or "image/tiff" in asset.get("type", "")
    ]
    return data_keys or list(assets)
