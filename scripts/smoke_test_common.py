"""Shared data pipeline steps for smoke tests."""

import logging
import sys
from pathlib import Path

import pandas as pd

from blue_frogs.data.downloader import build_download_manifest, download_batch
from blue_frogs.data.inat_client import (
    fetch_anura_observations_page,
    fetch_observation_metadata,
    get_photo_urls,
)
from blue_frogs.data.labels import download_womack_labels, filter_inat_records, load_womack_labels
from blue_frogs.data.splits import create_stratified_splits

logger = logging.getLogger(__name__)


def step_download_labels(data_dir: Path) -> pd.DataFrame:
    """Download and parse Womack labels."""
    csv_path = data_dir / "womack_labels.csv"
    if not csv_path.exists():
        logger.info("Downloading Womack labels from GitHub...")
        download_womack_labels(csv_path)
    else:
        logger.info(f"Using cached labels at {csv_path}")

    df = load_womack_labels(csv_path)
    inat_df = filter_inat_records(df)
    logger.info(f"Found {len(inat_df)} iNaturalist positive observations")
    return inat_df


def step_fetch_positive_metadata(positive_obs_ids: list[int]) -> list[dict]:
    """Fetch metadata for positive observations from iNat API."""
    logger.info(f"Fetching metadata for {len(positive_obs_ids)} positive observations...")
    metadata = fetch_observation_metadata(positive_obs_ids)
    logger.info(f"Retrieved metadata for {len(metadata)} positive observations")
    return metadata


def step_fetch_negative_metadata(
    positive_obs_ids: set[int],
    n_negatives: int = 2000,
    max_pages: int = 50,
) -> list[dict]:
    """Fetch random Anura observations as negatives from iNat API."""
    logger.info(f"Fetching negative observations from iNat API (target: {n_negatives})...")
    negatives = []
    id_above = 0

    for page_num in range(max_pages):
        if len(negatives) >= n_negatives:
            break
        response = fetch_anura_observations_page(id_above=id_above, per_page=200)
        results = response.get("results", [])
        if not results:
            break

        for obs in results:
            if obs["id"] not in positive_obs_ids and get_photo_urls(obs):
                negatives.append(obs)

        id_above = results[-1]["id"]
        logger.info(f"  Page {page_num + 1}: {len(negatives)} negatives collected so far")

    negatives = negatives[:n_negatives]
    logger.info(f"Collected {len(negatives)} negative observations")
    return negatives


def step_download_images(metadata: list[dict], image_dir: Path, max_workers: int = 4) -> dict:
    """Download images for all observations."""
    manifest = build_download_manifest(metadata, image_dir)
    logger.info(f"Download manifest: {len(manifest)} images")
    stats = download_batch(manifest, max_workers=max_workers)
    logger.info(f"Download stats: {stats}")
    return stats


def step_build_metadata(
    positive_metadata: list[dict],
    negative_metadata: list[dict],
    image_dir: Path,
) -> list[dict]:
    """Build flat metadata list for FrogDataset."""
    entries = []
    for obs in positive_metadata:
        obs_id = obs["id"]
        for obs_photo in obs.get("observation_photos", []):
            photo = obs_photo.get("photo", {})
            photo_id = photo.get("id")
            if not photo_id:
                continue
            photo_path = f"{obs_id}/{photo_id}.jpg"
            full_path = image_dir / photo_path
            if full_path.exists():
                entries.append({
                    "observation_id": obs_id,
                    "photo_id": photo_id,
                    "photo_path": photo_path,
                    "label": 1,
                })

    for obs in negative_metadata:
        obs_id = obs["id"]
        for obs_photo in obs.get("observation_photos", []):
            photo = obs_photo.get("photo", {})
            photo_id = photo.get("id")
            if not photo_id:
                continue
            photo_path = f"{obs_id}/{photo_id}.jpg"
            full_path = image_dir / photo_path
            if full_path.exists():
                entries.append({
                    "observation_id": obs_id,
                    "photo_id": photo_id,
                    "photo_path": photo_path,
                    "label": 0,
                })

    n_pos = sum(1 for e in entries if e["label"] == 1)
    n_neg = sum(1 for e in entries if e["label"] == 0)
    logger.info(f"Built metadata: {n_pos} positive, {n_neg} negative images")
    return entries


def step_split(metadata: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split."""
    df = pd.DataFrame(metadata)
    train_df, val_df = create_stratified_splits(df, test_fraction=0.2, seed=seed)
    logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")
    train_meta = train_df.to_dict("records")
    val_meta = val_df.to_dict("records")
    return train_meta, val_meta


def run_shared_data_pipeline(
    config: dict, data_dir: Path
) -> tuple[list[dict], list[dict], list[dict], list[dict], Path]:
    """Run the full shared data pipeline.

    Returns (entries, train_meta, val_meta, all_metadata, image_dir).
    """
    image_dir = data_dir / "raw_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    inat_df = step_download_labels(data_dir)
    positive_obs_ids = inat_df["observation_id"].tolist()
    positive_obs_ids_set = set(positive_obs_ids)

    positive_metadata = step_fetch_positive_metadata(positive_obs_ids)

    n_negatives = config["data"]["n_negatives"]
    max_pages = config["smoke_test"]["max_negative_pages"]
    negative_metadata = step_fetch_negative_metadata(
        positive_obs_ids_set, n_negatives=n_negatives, max_pages=max_pages
    )

    all_metadata = positive_metadata + negative_metadata
    step_download_images(all_metadata, image_dir, max_workers=4)

    entries = step_build_metadata(positive_metadata, negative_metadata, image_dir)
    if not entries:
        logger.error("No images found after download. Exiting.")
        sys.exit(1)

    train_meta, val_meta = step_split(entries, seed=config["data"]["seed"])
    return entries, train_meta, val_meta, all_metadata, image_dir


def print_results(
    model_name: str,
    entries: list[dict],
    train_meta: list[dict],
    val_meta: list[dict],
    config: dict,
    metrics: dict,
):
    """Print smoke test summary."""
    n_pos = sum(1 for e in entries if e["label"] == 1)
    n_neg = sum(1 for e in entries if e["label"] == 0)
    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST RESULTS — {model_name}")
    print(f"{'=' * 60}")
    print(f"Dataset: {n_pos} positive, {n_neg} negative images")
    print(f"Train: {len(train_meta)}, Val: {len(val_meta)}")
    print(f"AUROC:  {metrics['auroc']:.4f}")
    print(f"AUPRC:  {metrics['auprc']:.4f}")
    print(f"F1:     {metrics['f1_optimal']:.4f} (threshold={metrics['optimal_threshold']:.4f})")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"{'=' * 60}")
    print(f"\nSmoke test {'PASSED' if metrics['auroc'] > 0 else 'FAILED'}")


def save_metrics(metrics: dict, output_dir: Path, filename: str = "smoke_test_metrics.json"):
    """Save metrics dict to JSON."""
    import json
    metrics_path = output_dir / filename
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {metrics_path}")
