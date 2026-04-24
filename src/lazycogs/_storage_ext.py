"""STAC Storage Extension metadata parsing for obstore kwargs inference."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _storage_extension_version(stac_extensions: list[str]) -> str | None:
    """Return the storage extension version string, or ``None`` if absent.

    Parses the version from a URL like
    ``https://stac-extensions.github.io/storage/v1.0.0/schema.json``.
    """
    for url in stac_extensions:
        if "stac-extensions.github.io/storage" in url:
            # strip trailing /schema.json then take the last path component
            version = url.removesuffix("/schema.json").rsplit("/", 1)[-1].lstrip("v")
            return version
    return None


def _extract_store_kwargs_v1(
    item: dict[str, Any], asset: dict[str, Any]
) -> dict[str, Any]:
    """Extract obstore kwargs from a STAC Storage Extension v1.0.0 item.

    Asset-level fields take precedence over item ``properties``-level fields.
    Only ``region`` and ``requester_pays`` are mapped; ``tier`` has no obstore
    equivalent and is ignored.
    """
    props = item.get("properties", {})
    platform = (
        asset.get("storage:platform") or props.get("storage:platform", "")
    ).upper()
    region = asset.get("storage:region") or props.get("storage:region")
    requester_pays = asset.get(
        "storage:requester_pays", props.get("storage:requester_pays", False)
    )

    kwargs: dict[str, Any] = {}
    if platform == "AWS":
        if region:
            kwargs["region"] = region
        if requester_pays:
            kwargs["request_payer"] = True
    return kwargs


def _extract_store_kwargs_v2(
    item: dict[str, Any], asset: dict[str, Any]
) -> dict[str, Any]:
    """Extract obstore kwargs from a STAC Storage Extension v2.0.0 item.

    Resolves ``storage:refs`` on the asset against ``storage:schemes`` in item
    properties.  Uses the first matching scheme.  Only ``region``,
    ``requester_pays``, and custom S3 endpoints are mapped.
    """
    schemes: dict[str, Any] = item.get("properties", {}).get("storage:schemes", {})
    refs: list[str] = asset.get("storage:refs", [])
    scheme = next((schemes[r] for r in refs if r in schemes), None)
    if scheme is None:
        return {}

    store_type: str = scheme.get("type", "")
    kwargs: dict[str, Any] = {}

    if "s3" in store_type:
        if region := scheme.get("region"):
            kwargs["region"] = region
        if scheme.get("requester_pays"):
            kwargs["request_payer"] = True
        if store_type == "custom-s3":
            platform: str = scheme.get("platform", "")
            # only treat platform as a concrete endpoint when it contains no
            # URI template variables (e.g. {region})
            if platform and "{" not in platform:
                kwargs["endpoint"] = platform

    return kwargs


def _extract_store_kwargs(
    item: dict[str, Any], asset: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch storage extension kwarg extraction by schema version.

    Returns an empty dict when the storage extension is absent, the version is
    unrecognised, or parsing fails.
    """
    version = _storage_extension_version(item.get("stac_extensions", []))
    if version is None:
        return {}
    major = version.split(".")[0]
    if major == "1":
        return _extract_store_kwargs_v1(item, asset)
    if major == "2":
        return _extract_store_kwargs_v2(item, asset)
    logger.debug("Unrecognised storage extension version %r — skipping", version)
    return {}
