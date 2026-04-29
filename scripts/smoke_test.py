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
