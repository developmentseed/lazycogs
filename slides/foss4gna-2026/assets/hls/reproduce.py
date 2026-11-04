"""Fetch the current 2025 HLSS30 GeoParquet partitions via their Iceberg manifest."""

# ruff: noqa: INP001

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import requests
from fastavro import reader

LOGGER = logging.getLogger(__name__)
BUCKET = "nasa-maap-data-store"
PREFIX = "file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0"
METADATA_URL = (
    f"https://s3.us-west-2.amazonaws.com/{BUCKET}/{PREFIX}"
    "/iceberg/metadata/latest.metadata.json"
)
CACHE = Path(__file__).resolve().parents[4] / "docs/notebooks/data"
MAX_DOWNLOAD_BYTES = 700_000_000
EXPECTED_MONTHS = 12
DELETED_STATUS = 2


def fetch(url: str) -> bytes:
    """Fetch a public archive object over HTTPS without request signing."""
    base_url = f"https://s3.us-west-2.amazonaws.com/{BUCKET}/"
    if not url.startswith(base_url):
        raise ValueError("Archive fetch URL must use the documented HTTPS bucket")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def main() -> None:
    """Download only the 2025 HLSS30 monthly files named by the current snapshot."""
    metadata = json.loads(fetch(METADATA_URL))
    snapshot = next(
        item
        for item in metadata["snapshots"]
        if item["snapshot-id"] == metadata["current-snapshot-id"]
    )
    manifest_list = fetch(
        snapshot["manifest-list"].replace(
            f"s3://{BUCKET}/",
            f"https://s3.us-west-2.amazonaws.com/{BUCKET}/",
        ),
    )
    manifest_entries = list(reader(io.BytesIO(manifest_list)))
    manifest_paths = [
        item["manifest_path"] for item in manifest_entries if item["content"] == 0
    ]

    files = []
    for manifest_path in manifest_paths:
        manifest_url = manifest_path.replace(
            f"s3://{BUCKET}/",
            f"https://s3.us-west-2.amazonaws.com/{BUCKET}/",
        )
        entries = reader(io.BytesIO(fetch(manifest_url)))
        files.extend(
            entry["data_file"]
            for entry in entries
            if entry["status"] != DELETED_STATUS
            and f"{PREFIX}/year=2025/" in entry["data_file"]["file_path"]
        )

    total_size = sum(item["file_size_in_bytes"] for item in files)
    if len(files) != EXPECTED_MONTHS or total_size > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_MONTHS} 2025 monthly partitions within "
            f"{MAX_DOWNLOAD_BYTES} bytes; "
            f"snapshot lists {len(files)} files / {total_size} bytes",
        )

    for item in files:
        source = item["file_path"].replace(
            f"s3://{BUCKET}/",
            f"https://s3.us-west-2.amazonaws.com/{BUCKET}/",
        )
        relative = item["file_path"].split(f"{PREFIX}/", 1)[1]
        destination = CACHE / "hls-stac-geoparquet-archive/v2/HLSS30_2.0" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.exists()
            and destination.stat().st_size == item["file_size_in_bytes"]
        ):
            LOGGER.info(
                "Using cached %s (%d bytes)",
                destination.name,
                destination.stat().st_size,
            )
            continue
        payload = fetch(source)
        if len(payload) != item["file_size_in_bytes"]:
            raise RuntimeError(
                f"Unexpected object length for {destination.name}: {len(payload)}",
            )
        destination.write_bytes(payload)
        LOGGER.info("Fetched %s (%d bytes)", destination.name, len(payload))

    LOGGER.info(
        "Prepared 2025 HLSS30 index only: %d files, %d bytes, %d records",
        len(files),
        total_size,
        sum(item["record_count"] for item in files),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
