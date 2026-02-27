"""iNaturalist API client for fetching observation metadata and photo URLs."""

import re
import time
from typing import Any

from pyinaturalist import get_observations

from blue_frogs.config import ANURA_TAXON_ID, INAT_RATE_LIMIT

_SIZE_PATTERN = re.compile(r"/(square|small|thumb|large|original)\.")


def build_medium_url(photo_url: str) -> str:
    """Convert any iNaturalist photo URL to medium size (1024px longest side)."""
    return _SIZE_PATTERN.sub("/medium.", photo_url)


def get_photo_urls(observation: dict) -> list[str]:
    """Extract medium-resolution photo URLs from an observation dict."""
    urls = []
    for obs_photo in observation.get("observation_photos", []):
        photo = obs_photo.get("photo", {})
        url = photo.get("url")
        if url:
            urls.append(build_medium_url(url))
    return urls


def fetch_observation_metadata(
    observation_ids: list[int],
    batch_size: int = 30,
) -> list[dict[str, Any]]:
    """Fetch metadata for a list of observation IDs from the iNaturalist API.

    Batches requests to respect rate limits.
    """
    results = []
    for i in range(0, len(observation_ids), batch_size):
        batch = observation_ids[i : i + batch_size]
        response = get_observations(id=batch, per_page=batch_size)
        results.extend(response.get("results", []))
        # Rate limiting
        if i + batch_size < len(observation_ids):
            time.sleep(60 / INAT_RATE_LIMIT)
    return results


def fetch_anura_observations_page(
    id_above: int = 0,
    per_page: int = 200,
) -> dict[str, Any]:
    """Fetch a single page of research-grade Anura observations.

    Use id_above for cursor-based pagination through the full corpus.
    """
    return get_observations(
        taxon_id=ANURA_TAXON_ID,
        quality_grade="research",
        photos=True,
        per_page=per_page,
        id_above=id_above,
        order_by="id",
        order="asc",
    )
