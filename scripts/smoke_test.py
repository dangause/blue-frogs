"""End-to-end smoke test: download data, train Model A, evaluate."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from blue_frogs.config import DATA_DIR, MODEL_DIR, RANDOM_SEED
from blue_frogs.data.dataset import FrogDataset, get_train_transforms, get_val_transforms
from blue_frogs.data.downloader import build_download_manifest, download_batch
from blue_frogs.data.inat_client import (
    fetch_anura_observations_page,
    fetch_observation_metadata,
    get_photo_urls,
)
from blue_frogs.data.labels import download_womack_labels, filter_inat_records, load_womack_labels
from blue_frogs.data.splits import create_stratified_splits
from blue_frogs.evaluation.metrics import compute_classification_metrics
from blue_frogs.models.common import make_weighted_sampler
from blue_frogs.models.model_a import EfficientNetClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def step_download_labels(data_dir: Path) -> pd.DataFrame:
    """Step 1: Download and parse Womack labels."""
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


def step_fetch_positive_metadata(
    positive_obs_ids: list[int],
) -> list[dict]:
    """Step 2: Fetch metadata for positive observations from iNat API."""
    logger.info(f"Fetching metadata for {len(positive_obs_ids)} positive observations...")
    metadata = fetch_observation_metadata(positive_obs_ids)
    logger.info(f"Retrieved metadata for {len(metadata)} positive observations")
    return metadata


def step_fetch_negative_metadata(
    positive_obs_ids: set[int],
    n_negatives: int = 2000,
    max_pages: int = 50,
) -> list[dict]:
    """Step 3: Fetch random Anura observations as negatives from iNat API."""
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
        logger.info(
            f"  Page {page_num + 1}: {len(negatives)} negatives collected so far"
        )

    negatives = negatives[:n_negatives]
    logger.info(f"Collected {len(negatives)} negative observations")
    return negatives


def step_download_images(
    metadata: list[dict], image_dir: Path, max_workers: int = 4
) -> dict:
    """Step 4: Download images for all observations."""
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
    """Step 5: Build flat metadata list for FrogDataset."""
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


def step_split(
    metadata: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Step 6: Stratified train/val split."""
    df = pd.DataFrame(metadata)
    train_df, val_df = create_stratified_splits(df, test_fraction=0.2, seed=seed)
    logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")
    train_meta = train_df.to_dict("records")
    val_meta = val_df.to_dict("records")
    return train_meta, val_meta


def step_train(
    train_meta: list[dict],
    val_meta: list[dict],
    image_dir: Path,
    config: dict,
    output_dir: Path,
) -> str:
    """Step 7: Train Model A."""
    pl.seed_everything(RANDOM_SEED)

    train_dataset = FrogDataset(train_meta, image_dir, transform=get_train_transforms())
    val_dataset = FrogDataset(val_meta, image_dir, transform=get_val_transforms())

    train_labels = [m["label"] for m in train_meta]
    sampler = make_weighted_sampler(train_labels)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=False,
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
        dirpath=output_dir,
        filename="best-{val/auprc:.4f}",
        monitor="val/auprc",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc",
        mode="max",
        patience=config["training"]["early_stopping_patience"],
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=False,  # no W&B for smoke test
        deterministic=True,
        enable_progress_bar=True,
    )

    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Training complete. Best checkpoint: {checkpoint_cb.best_model_path}")
    return checkpoint_cb.best_model_path


def step_evaluate(
    best_checkpoint: str,
    val_meta: list[dict],
    image_dir: Path,
    config: dict,
) -> dict:
    """Step 8: Evaluate best model on validation set."""
    model = EfficientNetClassifier.load_from_checkpoint(best_checkpoint)
    model.eval()

    val_dataset = FrogDataset(val_meta, image_dir, transform=get_val_transforms())
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
    )

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_score = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = compute_classification_metrics(y_true, y_score)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "smoke_test.yaml",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "smoke_test")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    image_dir = args.data_dir / "raw_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Labels
    inat_df = step_download_labels(args.data_dir)
    positive_obs_ids = inat_df["observation_id"].tolist()
    positive_obs_ids_set = set(positive_obs_ids)

    # Step 2: Positive metadata
    positive_metadata = step_fetch_positive_metadata(positive_obs_ids)

    # Step 3: Negative metadata
    n_negatives = config["data"]["n_negatives"]
    max_pages = config["smoke_test"]["max_negative_pages"]
    negative_metadata = step_fetch_negative_metadata(
        positive_obs_ids_set, n_negatives=n_negatives, max_pages=max_pages
    )

    # Step 4: Download images
    all_metadata = positive_metadata + negative_metadata
    step_download_images(all_metadata, image_dir, max_workers=4)

    # Step 5: Build metadata
    entries = step_build_metadata(positive_metadata, negative_metadata, image_dir)
    if not entries:
        logger.error("No images found after download. Exiting.")
        sys.exit(1)

    # Step 6: Split
    train_meta, val_meta = step_split(entries, seed=config["data"]["seed"])

    # Step 7: Train
    best_ckpt = step_train(train_meta, val_meta, image_dir, config, args.output_dir)

    # Step 8: Evaluate
    metrics = step_evaluate(best_ckpt, val_meta, image_dir, config)

    # Print summary
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    n_pos = sum(1 for e in entries if e["label"] == 1)
    n_neg = sum(1 for e in entries if e["label"] == 0)
    print(f"Dataset: {n_pos} positive, {n_neg} negative images")
    print(f"Train: {len(train_meta)}, Val: {len(val_meta)}")
    print(f"Epochs: {config['training']['max_epochs']}")
    print(f"AUROC:  {metrics['auroc']:.4f}")
    print(f"AUPRC:  {metrics['auprc']:.4f}")
    print(f"F1:     {metrics['f1_optimal']:.4f} (threshold={metrics['optimal_threshold']:.4f})")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print("=" * 60)

    # Save metrics
    metrics_path = args.output_dir / "smoke_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {metrics_path}")

    print("\nSmoke test PASSED" if metrics["auroc"] > 0 else "\nSmoke test FAILED")


if __name__ == "__main__":
    main()
