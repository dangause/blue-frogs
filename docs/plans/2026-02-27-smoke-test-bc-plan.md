# Model B & C Smoke Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the full Model B (Fusion) and Model C (DINOv2) pipelines end-to-end with real data, reusing the same dataset from the Model A smoke test.

**Architecture:** Extract shared data pipeline steps into `scripts/smoke_test_common.py`, then create thin per-model scripts `scripts/smoke_test_b.py` and `scripts/smoke_test_c.py`. Model B adds a YOLO crop preprocessing step and a custom dataset that computes LAB color features. Model C implements two-stage training (linear probe → fine-tune). Both produce metrics JSON comparable to Model A.

**Tech Stack:** Python 3.11, PyTorch (CPU), PyTorch Lightning, ultralytics (YOLO), Docker

---

### Task 1: Extract shared steps into smoke_test_common.py

**Files:**
- Create: `scripts/smoke_test_common.py`
- Modify: `scripts/smoke_test.py`

**Step 1: Create the shared module**

Extract these functions from `smoke_test.py` into `smoke_test_common.py` with no changes to their signatures or implementations:

```python
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


def run_shared_data_pipeline(config: dict, data_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict], Path]:
    """Run the full shared data pipeline. Returns (entries, train_meta, val_meta, all_metadata, image_dir)."""
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


def print_results(model_name: str, entries: list[dict], train_meta: list[dict], val_meta: list[dict], config: dict, metrics: dict):
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
```

**Step 2: Refactor smoke_test.py to use the shared module**

Replace `scripts/smoke_test.py` with a thin wrapper that imports from `smoke_test_common`:

```python
"""End-to-end smoke test: download data, train Model A, evaluate."""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from blue_frogs.config import DATA_DIR, MODEL_DIR, RANDOM_SEED
from blue_frogs.data.dataset import FrogDataset, get_train_transforms, get_val_transforms
from blue_frogs.evaluation.metrics import compute_classification_metrics
from blue_frogs.models.common import make_weighted_sampler
from blue_frogs.models.model_a import EfficientNetClassifier

from smoke_test_common import run_shared_data_pipeline, print_results, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def step_train(train_meta, val_meta, image_dir, config, output_dir):
    """Train Model A."""
    pl.seed_everything(RANDOM_SEED)

    train_dataset = FrogDataset(train_meta, image_dir, transform=get_train_transforms())
    val_dataset = FrogDataset(val_meta, image_dir, transform=get_val_transforms())

    train_labels = [m["label"] for m in train_meta]
    sampler = make_weighted_sampler(train_labels)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=config["training"].get("num_workers", 0), pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config["training"].get("num_workers", 0), pin_memory=False,
    )

    model = EfficientNetClassifier(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        freeze_backbone=config["model"]["freeze_backbone"],
        loss_type=config["training"]["loss_type"],
        pos_weight=config["training"]["pos_weight"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        optimizer=config["training"]["optimizer"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc", mode="max",
        patience=config["training"]["early_stopping_patience"],
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"], accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb], logger=False,
        deterministic=True, enable_progress_bar=True,
    )

    logger.info("Starting Model A training...")
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Training complete. Best checkpoint: {checkpoint_cb.best_model_path}")
    return checkpoint_cb.best_model_path


def step_evaluate(best_checkpoint, val_meta, image_dir, config):
    """Evaluate best Model A on validation set."""
    model = EfficientNetClassifier.load_from_checkpoint(best_checkpoint)
    model.eval()

    val_dataset = FrogDataset(val_meta, image_dir, transform=get_val_transforms())
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["training"].get("num_workers", 0),
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_score = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    return compute_classification_metrics(y_true, y_score)


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test — Model A")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "smoke_test.yaml")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "smoke_test")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    entries, train_meta, val_meta, _, image_dir = run_shared_data_pipeline(config, args.data_dir)

    best_ckpt = step_train(train_meta, val_meta, image_dir, config, args.output_dir)
    metrics = step_evaluate(best_ckpt, val_meta, image_dir, config)

    print_results("Model A (EfficientNetV2-S)", entries, train_meta, val_meta, config, metrics)
    save_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    main()
```

**Step 3: Verify Model A still works**

Run inside Docker:
```bash
docker build -t blue-frogs-smoke . && docker run --rm blue-frogs-smoke --help
```

Expected: Shows argparse help for `smoke_test.py`.

**Step 4: Commit**

```bash
git add scripts/smoke_test_common.py scripts/smoke_test.py
git commit -m "Extract shared data pipeline into smoke_test_common.py"
```

---

### Task 2: Create smoke_test_b.yaml config

**Files:**
- Create: `configs/smoke_test_b.yaml`

**Step 1: Write the config**

```yaml
detector:
  model: yolov8m.pt
  conf_threshold: 0.5
  padding_fraction: 0.1

classifier:
  backbone: tf_efficientnetv2_s
  pretrained: true
  color_feature_dim: 30
  fusion_hidden: 256
  dropout: 0.3

training:
  loss_type: weighted_bce
  pos_weight: 20.0
  optimizer: sgd
  learning_rate: 0.001
  weight_decay: 0.0
  max_epochs: 3
  batch_size: 8
  num_workers: 0
  early_stopping_patience: 3
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

data:
  image_size: 384
  seed: 42
  n_negatives: 2000

smoke_test:
  max_negative_pages: 50
```

**Step 2: Commit**

```bash
git add configs/smoke_test_b.yaml
git commit -m "Add Model B smoke test config"
```

---

### Task 3: Create smoke_test_b.py

**Files:**
- Create: `scripts/smoke_test_b.py`

**Step 1: Write the Model B smoke test script**

```python
"""End-to-end smoke test: download data, YOLO crop, train Model B (Fusion), evaluate."""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from blue_frogs.config import DATA_DIR, MODEL_DIR, RANDOM_SEED
from blue_frogs.data.color_features import extract_lab_features
from blue_frogs.data.dataset import get_train_transforms, get_val_transforms
from blue_frogs.evaluation.metrics import compute_classification_metrics
from blue_frogs.models.common import make_weighted_sampler
from blue_frogs.models.model_b_classifier import FusionClassifier
from blue_frogs.models.model_b_detector import FrogDetector

from smoke_test_common import run_shared_data_pipeline, print_results, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def step_crop_images(
    metadata: list[dict],
    raw_image_dir: Path,
    crop_dir: Path,
    conf_threshold: float = 0.5,
    padding_fraction: float = 0.1,
):
    """Run YOLO frog detection and save crops to disk."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    detector = FrogDetector(conf_threshold=conf_threshold)

    skipped = 0
    cropped = 0
    failed = 0

    for entry in tqdm(metadata, desc="Cropping frogs"):
        photo_path = entry["photo_path"]
        crop_path = crop_dir / photo_path
        if crop_path.exists():
            skipped += 1
            continue

        raw_path = raw_image_dir / photo_path
        if not raw_path.exists():
            failed += 1
            continue

        image = cv2.imread(str(raw_path))
        if image is None:
            failed += 1
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        crop = detector.crop_frog(image_rgb, padding_fraction=padding_fraction)
        if crop is not None and crop.size > 0:
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)
            cropped += 1
        else:
            failed += 1

    logger.info(f"Crop stats: {cropped} cropped, {skipped} cached, {failed} failed")


class FrogDatasetWithColorFeatures(Dataset):
    """Dataset that returns (image_tensor, color_features, label) for Model B."""

    def __init__(self, metadata, image_dir, crop_dir, transform=None):
        self.metadata = metadata
        self.image_dir = Path(image_dir)
        self.crop_dir = Path(crop_dir)
        self.transform = transform or get_val_transforms()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        photo_path = entry["photo_path"]

        # Prefer crop, fall back to raw
        crop_path = self.crop_dir / photo_path
        raw_path = self.image_dir / photo_path
        img_path = crop_path if crop_path.exists() else raw_path

        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract LAB features from raw uint8 image BEFORE transforms
        color_feats = extract_lab_features(image_rgb)
        color_feats_tensor = torch.tensor(color_feats, dtype=torch.float32)

        # Apply albumentations transforms
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed["image"]

        label = entry["label"]
        return image_tensor, color_feats_tensor, label


def step_train(train_meta, val_meta, image_dir, crop_dir, config, output_dir):
    """Train Model B (FusionClassifier)."""
    pl.seed_everything(RANDOM_SEED)

    train_dataset = FrogDatasetWithColorFeatures(
        train_meta, image_dir, crop_dir, transform=get_train_transforms()
    )
    val_dataset = FrogDatasetWithColorFeatures(
        val_meta, image_dir, crop_dir, transform=get_val_transforms()
    )

    train_labels = [m["label"] for m in train_meta]
    sampler = make_weighted_sampler(train_labels)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=config["training"].get("num_workers", 0), pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config["training"].get("num_workers", 0), pin_memory=False,
    )

    clf = config["classifier"]
    model = FusionClassifier(
        backbone=clf["backbone"],
        pretrained=clf["pretrained"],
        color_feature_dim=clf["color_feature_dim"],
        fusion_hidden=clf["fusion_hidden"],
        dropout=clf["dropout"],
        loss_type=config["training"]["loss_type"],
        pos_weight=config["training"]["pos_weight"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        optimizer=config["training"]["optimizer"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc", mode="max",
        patience=config["training"]["early_stopping_patience"],
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"], accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb], logger=False,
        deterministic=True, enable_progress_bar=True,
    )

    logger.info("Starting Model B training...")
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Training complete. Best checkpoint: {checkpoint_cb.best_model_path}")
    return checkpoint_cb.best_model_path


def step_evaluate(best_checkpoint, val_meta, image_dir, crop_dir, config):
    """Evaluate best Model B on validation set."""
    model = FusionClassifier.load_from_checkpoint(best_checkpoint)
    model.eval()

    val_dataset = FrogDatasetWithColorFeatures(
        val_meta, image_dir, crop_dir, transform=get_val_transforms()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["training"].get("num_workers", 0),
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, color_feats, labels in val_loader:
            logits = model(images, color_feats)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_score = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    return compute_classification_metrics(y_true, y_score)


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test — Model B (Fusion)")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "smoke_test_b.yaml")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "smoke_test_b")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    entries, train_meta, val_meta, _, image_dir = run_shared_data_pipeline(config, args.data_dir)

    # Model B extra step: YOLO crop
    crop_dir = args.data_dir / "cropped_images"
    all_entries = entries  # crop all entries (train + val)
    det_cfg = config["detector"]
    step_crop_images(
        all_entries, image_dir, crop_dir,
        conf_threshold=det_cfg["conf_threshold"],
        padding_fraction=det_cfg["padding_fraction"],
    )

    best_ckpt = step_train(train_meta, val_meta, image_dir, crop_dir, config, args.output_dir)
    metrics = step_evaluate(best_ckpt, val_meta, image_dir, crop_dir, config)

    print_results("Model B (Fusion: EfficientNetV2-S + LAB)", entries, train_meta, val_meta, config, metrics)
    save_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/smoke_test_b.py
git commit -m "Add Model B smoke test script with YOLO crop and LAB features"
```

---

### Task 4: Create smoke_test_c.yaml config

**Files:**
- Create: `configs/smoke_test_c.yaml`

**Step 1: Write the config**

```yaml
model:
  backbone: dinov2
  model_size: base
  hidden_dim: 256
  dropout: 0.3

linear_probe:
  freeze_backbone: true
  max_epochs: 2
  learning_rate: 0.001

finetune:
  freeze_backbone: false
  max_epochs: 2
  backbone_lr_factor: 0.01

training:
  loss_type: weighted_bce
  pos_weight: 20.0
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  batch_size: 8
  num_workers: 0
  early_stopping_patience: 3
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

data:
  image_size: 384
  seed: 42
  n_negatives: 2000

smoke_test:
  max_negative_pages: 50
```

**Step 2: Commit**

```bash
git add configs/smoke_test_c.yaml
git commit -m "Add Model C smoke test config with two-stage training"
```

---

### Task 5: Create smoke_test_c.py

**Files:**
- Create: `scripts/smoke_test_c.py`

**Step 1: Write the Model C smoke test script**

```python
"""End-to-end smoke test: download data, train Model C (DINOv2) two-stage, evaluate."""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from blue_frogs.config import DATA_DIR, MODEL_DIR, RANDOM_SEED
from blue_frogs.data.dataset import FrogDataset, get_train_transforms, get_val_transforms
from blue_frogs.evaluation.metrics import compute_classification_metrics
from blue_frogs.models.common import make_weighted_sampler
from blue_frogs.models.model_c import FoundationModelClassifier

from smoke_test_common import run_shared_data_pipeline, print_results, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _make_loaders(train_meta, val_meta, image_dir, batch_size, num_workers):
    """Build train and val DataLoaders."""
    train_dataset = FrogDataset(train_meta, image_dir, transform=get_train_transforms())
    val_dataset = FrogDataset(val_meta, image_dir, transform=get_val_transforms())

    train_labels = [m["label"] for m in train_meta]
    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader


def step_train_two_stage(train_meta, val_meta, image_dir, config, output_dir):
    """Train Model C in two stages: linear probe then fine-tune."""
    pl.seed_everything(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    model_cfg = config["model"]

    base_kwargs = dict(
        backbone=model_cfg["backbone"],
        model_size=model_cfg["model_size"],
        pretrained=True,
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        loss_type=config["training"]["loss_type"],
        pos_weight=config["training"]["pos_weight"],
        weight_decay=config["training"]["weight_decay"],
    )

    # --- Stage 1: Linear probe (frozen backbone) ---
    logger.info("=== Stage 1: Linear Probe (frozen backbone) ===")
    probe_cfg = config["linear_probe"]

    model = FoundationModelClassifier(
        freeze_backbone=True,
        learning_rate=probe_cfg["learning_rate"],
        optimizer=config["training"]["optimizer"],
        **base_kwargs,
    )

    train_loader, val_loader = _make_loaders(train_meta, val_meta, image_dir, batch_size, num_workers)

    probe_dir = output_dir / "linear_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=probe_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=probe_cfg["max_epochs"], accelerator="auto",
        callbacks=[checkpoint_cb], logger=False,
        deterministic=True, enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)
    probe_ckpt = checkpoint_cb.best_model_path
    logger.info(f"Linear probe complete. Best: {probe_ckpt}")

    # --- Stage 2: Fine-tune (unfrozen backbone, differential LR) ---
    logger.info("=== Stage 2: Fine-tune (unfrozen backbone, differential LR) ===")
    ft_cfg = config["finetune"]

    # Load probe checkpoint weights into a new unfrozen model
    probe_model = FoundationModelClassifier.load_from_checkpoint(probe_ckpt)
    model = FoundationModelClassifier(
        freeze_backbone=False,
        learning_rate=config["training"]["learning_rate"],
        optimizer=config["training"]["optimizer"],
        **base_kwargs,
    )
    # Transfer learned head weights from probe
    model.head.load_state_dict(probe_model.head.state_dict())

    train_loader, val_loader = _make_loaders(train_meta, val_meta, image_dir, batch_size, num_workers)

    ft_dir = output_dir / "finetune"
    ft_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ft_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc", mode="max",
        patience=config["training"]["early_stopping_patience"],
    )

    trainer = pl.Trainer(
        max_epochs=ft_cfg["max_epochs"], accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb], logger=False,
        deterministic=True, enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)
    ft_ckpt = checkpoint_cb.best_model_path
    logger.info(f"Fine-tune complete. Best: {ft_ckpt}")
    return ft_ckpt


def step_evaluate(best_checkpoint, val_meta, image_dir, config):
    """Evaluate best Model C on validation set."""
    model = FoundationModelClassifier.load_from_checkpoint(best_checkpoint)
    model.eval()

    val_dataset = FrogDataset(val_meta, image_dir, transform=get_val_transforms())
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["training"].get("num_workers", 0),
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_score = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    return compute_classification_metrics(y_true, y_score)


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test — Model C (DINOv2)")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "smoke_test_c.yaml")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "smoke_test_c")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    entries, train_meta, val_meta, _, image_dir = run_shared_data_pipeline(config, args.data_dir)

    best_ckpt = step_train_two_stage(train_meta, val_meta, image_dir, config, args.output_dir)
    metrics = step_evaluate(best_ckpt, val_meta, image_dir, config)

    print_results("Model C (DINOv2 ViT-B/14)", entries, train_meta, val_meta, config, metrics)
    save_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/smoke_test_c.py
git commit -m "Add Model C smoke test script with two-stage DINOv2 training"
```

---

### Task 6: Update Dockerfile for Model B dependencies

**Files:**
- Modify: `Dockerfile`

**Step 1: Add ultralytics to the pip install**

In the builder stage, add `ultralytics` after the torch install line:

```dockerfile
# Install CPU-only PyTorch first, then the project, then extras
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir ultralytics
```

**Step 2: Build and verify all three scripts**

```bash
docker build -t blue-frogs-smoke .
docker run --rm blue-frogs-smoke --help
docker run --rm --entrypoint python blue-frogs-smoke scripts/smoke_test_b.py --help
docker run --rm --entrypoint python blue-frogs-smoke scripts/smoke_test_c.py --help
```

Expected: All three show argparse help.

**Step 3: Commit**

```bash
git add Dockerfile
git commit -m "Add ultralytics to Dockerfile for Model B YOLO support"
```

---

### Task 7: Run Model B smoke test

**Step 1: Run in Docker**

```bash
docker run --shm-size=2g -m 8g -v $(pwd)/data:/app/data \
  --entrypoint python blue-frogs-smoke \
  scripts/smoke_test_b.py --data-dir /app/data --output-dir /app/data/smoke_output_b
```

Expected progression:
1. Shared data pipeline (labels, metadata, download — mostly cached)
2. "Cropping frogs" progress bar
3. Training for 3 epochs
4. "SMOKE TEST RESULTS — Model B" with metrics
5. "Smoke test PASSED"

**Step 2: Verify metrics**

Check `data/smoke_output_b/smoke_test_metrics.json` exists with auroc, auprc, f1_optimal keys.

**Step 3: Commit any fixes**

If the run reveals issues, fix and commit.

---

### Task 8: Run Model C smoke test

**Step 1: Run in Docker**

```bash
docker run --shm-size=2g -m 8g -v $(pwd)/data:/app/data \
  --entrypoint python blue-frogs-smoke \
  scripts/smoke_test_c.py --data-dir /app/data --output-dir /app/data/smoke_output_c
```

Expected progression:
1. Shared data pipeline (mostly cached)
2. "Stage 1: Linear Probe" — 2 epochs
3. "Stage 2: Fine-tune" — 2 epochs
4. "SMOKE TEST RESULTS — Model C" with metrics
5. "Smoke test PASSED"

**Step 2: Verify metrics**

Check `data/smoke_output_c/smoke_test_metrics.json` exists with auroc, auprc, f1_optimal keys.

**Step 3: Commit any fixes**

If the run reveals issues, fix and commit.
