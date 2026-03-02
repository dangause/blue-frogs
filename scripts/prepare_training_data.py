"""Generate labels.json and splits.json from downloaded images and Womack CSV.

Scans data/raw_images/ for observation directories, cross-references the Womack
et al. CSV for positive axanthism labels, and writes training metadata files.

Usage:
    python scripts/prepare_training_data.py --data-dir data/ --labels-csv data/womack_labels.csv
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from blue_frogs.config import RANDOM_SEED, TEST_FRACTION
from blue_frogs.data.labels import load_womack_labels, filter_inat_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_labels(image_dir: Path, labels_csv: Path) -> list[dict]:
    """Build a flat list of {observation_id, photo_id, photo_path, label} entries.

    Scans image_dir for observation subdirectories containing .jpg files.
    Cross-references Womack CSV to assign label=1 to positive observations.
    """
    # Parse positive observation IDs from Womack CSV
    womack_df = load_womack_labels(labels_csv)
    inat_df = filter_inat_records(womack_df)
    positive_obs_ids = set(inat_df["observation_id"].tolist())
    logger.info(f"Womack positives: {len(positive_obs_ids)} iNat observations")

    entries = []
    matched_positives = set()

    for obs_dir in sorted(image_dir.iterdir()):
        if not obs_dir.is_dir():
            continue

        photos = sorted(obs_dir.glob("*.jpg"))
        if not photos:
            logger.debug(f"Skipping empty observation dir: {obs_dir.name}")
            continue

        try:
            obs_id = int(obs_dir.name)
        except ValueError:
            logger.warning(f"Skipping non-numeric dir: {obs_dir.name}")
            continue

        label = 1 if obs_id in positive_obs_ids else 0
        if label == 1:
            matched_positives.add(obs_id)

        for photo_path in photos:
            photo_id = int(photo_path.stem)
            entries.append({
                "observation_id": obs_id,
                "photo_id": photo_id,
                "photo_path": f"{obs_id}/{photo_path.name}",
                "label": label,
            })

    missing = positive_obs_ids - matched_positives
    if missing:
        logger.warning(f"{len(missing)} Womack positives not found on disk: {missing}")

    n_pos = sum(1 for e in entries if e["label"] == 1)
    n_neg = sum(1 for e in entries if e["label"] == 0)
    logger.info(f"Built labels: {len(entries)} photos ({n_pos} positive, {n_neg} negative)")

    return entries


def build_splits(
    labels: list[dict], test_fraction: float = TEST_FRACTION, seed: int = RANDOM_SEED
) -> dict:
    """Generate train/test split indices, splitting by observation to prevent leakage.

    Returns dict with train_indices and test_indices (photo-level).
    """
    from blue_frogs.data.splits import create_stratified_splits

    # Build observation-level DataFrame for splitting
    obs_data = {}
    for i, entry in enumerate(labels):
        obs_id = entry["observation_id"]
        if obs_id not in obs_data:
            obs_data[obs_id] = {"observation_id": obs_id, "label": entry["label"], "indices": []}
        obs_data[obs_id]["indices"].append(i)

    obs_df = pd.DataFrame([
        {"observation_id": v["observation_id"], "label": v["label"]}
        for v in obs_data.values()
    ])

    train_obs_df, test_obs_df = create_stratified_splits(obs_df, test_fraction, seed)

    train_obs_ids = set(train_obs_df["observation_id"])
    test_obs_ids = set(test_obs_df["observation_id"])

    train_indices = [i for i, e in enumerate(labels) if e["observation_id"] in train_obs_ids]
    test_indices = [i for i, e in enumerate(labels) if e["observation_id"] in test_obs_ids]

    logger.info(
        f"Split: {len(train_indices)} train photos ({len(train_obs_ids)} obs), "
        f"{len(test_indices)} test photos ({len(test_obs_ids)} obs)"
    )

    return {"train_indices": train_indices, "test_indices": test_indices}


def main():
    parser = argparse.ArgumentParser(description="Generate training data files")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data directory containing raw_images/")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="Path to Womack et al. CSV")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    image_dir = args.data_dir / "raw_images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    labels = build_labels(image_dir, args.labels_csv)
    splits = build_splits(labels, args.test_fraction, args.seed)

    labels_path = args.data_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info(f"Saved {labels_path}")

    splits_path = args.data_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved {splits_path}")


if __name__ == "__main__":
    main()
