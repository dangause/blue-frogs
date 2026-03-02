# Data Preparation & HPC Runbook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a data preparation script that generates `labels.json` and `splits.json` from existing downloaded images, plus an HPC deployment runbook.

**Architecture:** The script scans `data/raw_images/`, cross-references `womack_labels.csv` for positive labels, and writes two JSON files. The runbook is a standalone markdown doc with copy-pasteable commands.

**Tech Stack:** Python (existing `labels.py`, `splits.py` modules), Markdown

---

### Task 1: Create data preparation script with tests

**Files:**
- Create: `scripts/prepare_training_data.py`
- Create: `tests/test_prepare_training_data.py`

**Step 1: Write failing tests**

Create `tests/test_prepare_training_data.py`:

```python
"""Tests for training data preparation script."""

import json

import numpy as np
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a minimal mock dataset with images and a Womack-style CSV."""
    image_dir = tmp_path / "raw_images"

    # Create 5 positive observations (matching Womack CSV format)
    positive_obs = [100456051, 75584907, 19547423, 12345678, 87654321]
    for obs_id in positive_obs:
        obs_dir = image_dir / str(obs_id)
        obs_dir.mkdir(parents=True)
        # 1-2 photos per observation
        for photo_idx in range(1, 3):
            photo_id = obs_id * 10 + photo_idx
            img = Image.new("RGB", (100, 100), color=(50, 100, 200))
            img.save(obs_dir / f"{photo_id}.jpg")

    # Create 20 negative observations (not in Womack CSV)
    for i in range(20):
        obs_id = 9000000 + i
        obs_dir = image_dir / str(obs_id)
        obs_dir.mkdir(parents=True)
        photo_id = obs_id * 10 + 1
        img = Image.new("RGB", (100, 100), color=(0, 128, 0))
        img.save(obs_dir / f"{photo_id}.jpg")

    # Create one empty observation dir (should be skipped)
    (image_dir / "9999999").mkdir(parents=True)

    # Create Womack-style CSV
    csv_path = tmp_path / "womack_labels.csv"
    lines = [
        "Source,Family,Genera,Species,Individuals Observed,Location,Longitude,Latitude,Country,Usual Pattern,Unusual Pattern,Pattern Notes,Iris color,Lifestage,Citation,Year,Notes",
    ]
    for obs_id in positive_obs:
        lines.append(
            f'iNat,Hylidae,Agalychnis,Agalychnis callidryas,1,Costa Rica,10.0,-85.0,Costa Rica,Green,Blue,Whole body,N/A,Adult,https://www.inaturalist.org/observations/{obs_id},2021,'
        )
    # Add one non-iNat record (should be ignored)
    lines.append(
        'Sci Lit,Hylidae,Acris,Acris crepitans,1,Virginia,-78.0,37.0,USA,,dorsal axanthic,,NA,juvenile,Volume 44 Issue,,',
    )
    csv_path.write_text("\n".join(lines))

    return tmp_path, csv_path


def test_build_labels_json(mock_dataset):
    """build_labels produces correct label assignments."""
    from scripts.prepare_training_data import build_labels

    data_dir, csv_path = mock_dataset
    labels = build_labels(data_dir / "raw_images", csv_path)

    # 5 positive obs * 2 photos + 20 negative obs * 1 photo = 30
    assert len(labels) == 30

    positives = [l for l in labels if l["label"] == 1]
    negatives = [l for l in labels if l["label"] == 0]
    assert len(positives) == 10  # 5 obs * 2 photos
    assert len(negatives) == 20

    # Check structure
    for entry in labels:
        assert "observation_id" in entry
        assert "photo_id" in entry
        assert "photo_path" in entry
        assert "label" in entry
        assert "/" in entry["photo_path"]  # obs_id/photo_id.jpg format


def test_build_splits_by_observation(mock_dataset):
    """build_splits splits by observation, not photo, preventing leakage."""
    from scripts.prepare_training_data import build_labels, build_splits

    data_dir, csv_path = mock_dataset
    labels = build_labels(data_dir / "raw_images", csv_path)
    splits = build_splits(labels, test_fraction=0.2, seed=42)

    train_indices = splits["train_indices"]
    test_indices = splits["test_indices"]

    assert len(train_indices) + len(test_indices) == len(labels)
    assert set(train_indices) & set(test_indices) == set()

    # No observation appears in both train and test
    train_obs = {labels[i]["observation_id"] for i in train_indices}
    test_obs = {labels[i]["observation_id"] for i in test_indices}
    assert train_obs & test_obs == set()
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prepare_training_data.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement prepare_training_data.py**

Create `scripts/prepare_training_data.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prepare_training_data.py -v`
Expected: 2 PASSED

**Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest`
Expected: All 52+ tests PASS

**Step 6: Run linter**

Run: `.venv/bin/ruff check scripts/prepare_training_data.py tests/test_prepare_training_data.py`
Expected: No errors

**Step 7: Commit**

```bash
git add scripts/prepare_training_data.py tests/test_prepare_training_data.py
git commit -m "Add data preparation script to generate labels.json and splits.json"
```

---

### Task 2: Write HPC deployment runbook

**Files:**
- Create: `docs/hpc-runbook.md`

**Step 1: Write the runbook**

Create `docs/hpc-runbook.md` with step-by-step deployment commands covering:
local data prep, Docker build/push, cluster data transfer, SIF pull, job submission, monitoring, and result retrieval. Reference actual file paths, config defaults, and environment variables from the SLURM scripts.

Key content sections:

1. **Prerequisites** — what must exist before starting
2. **Step 1: Prepare training data** — run prepare_training_data.py locally
3. **Step 2: Build GPU Docker image** — docker build + push
4. **Step 3: Transfer data to cluster** — rsync/scp commands
5. **Step 4: Pull container on cluster** — apptainer pull
6. **Step 5: Submit training jobs** — submit_all.sh with env var overrides
7. **Step 6: Monitor progress** — squeue, tail logs
8. **Step 7: Retrieve results** — scp comparison_summary back
9. **Troubleshooting** — common issues (OOM, missing SIF, YOLO weights)

**Step 2: Commit**

```bash
git add docs/hpc-runbook.md
git commit -m "Add HPC deployment runbook"
```

---

### Task 3: Final verification

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 2: Run linter on new files**

Run: `.venv/bin/ruff check scripts/prepare_training_data.py tests/test_prepare_training_data.py`
Expected: Clean

**Step 3: Push and update PR**

```bash
git push
```
