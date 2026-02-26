"""Download images from iNaturalist."""

import logging
import time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from blue_frogs.data.inat_client import build_medium_url

logger = logging.getLogger(__name__)


def build_download_manifest(
    observations: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Build a list of (url, save_path) entries from observation metadata."""
    manifest = []
    for obs in observations:
        obs_id = obs["id"]
        obs_dir = output_dir / str(obs_id)
        for obs_photo in obs.get("observation_photos", []):
            photo = obs_photo.get("photo", {})
            photo_id = photo.get("id")
            url = photo.get("url")
            if not url or not photo_id:
                continue
            medium_url = build_medium_url(url)
            ext = "jpg"
            save_path = obs_dir / f"{photo_id}.{ext}"
            manifest.append({
                "observation_id": obs_id,
                "photo_id": photo_id,
                "url": medium_url,
                "save_path": save_path,
            })
    return manifest


def download_image_sync(url: str, save_path: Path, timeout: int = 30) -> bool:
    """Download a single image. Returns True on success."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            return True  # skip already downloaded
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        save_path.write_bytes(response.content)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def download_batch(
    manifest: list[dict[str, Any]],
    max_workers: int = 8,
    rate_limit: float = 60,
) -> dict[str, int]:
    """Download images in parallel with rate limiting.

    Returns dict with counts: {"success": N, "failed": N, "skipped": N}.
    """
    stats = {"success": 0, "failed": 0, "skipped": 0}
    delay = 1.0 / rate_limit if rate_limit > 0 else 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for entry in manifest:
            if entry["save_path"].exists():
                stats["skipped"] += 1
                continue
            future = executor.submit(
                download_image_sync, entry["url"], entry["save_path"]
            )
            futures[future] = entry
            time.sleep(delay)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if future.result():
                stats["success"] += 1
            else:
                stats["failed"] += 1

    return stats
