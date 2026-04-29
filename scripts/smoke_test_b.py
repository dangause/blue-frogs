"""End-to-end smoke test: download data, YOLO crop, train Model B (Fusion), evaluate."""

import argparse
import logging
from pathlib import Path

import cv2
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
