"""Unified training script for all model architectures."""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from blue_frogs.config import MODEL_DIR, RANDOM_SEED
from blue_frogs.data.dataset import FrogDataset, get_train_transforms, get_val_transforms
from blue_frogs.data.splits import get_fold_indices
from blue_frogs.models.common import make_weighted_sampler
from blue_frogs.models.model_a import EfficientNetClassifier
from blue_frogs.models.model_c import FoundationModelClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "model_a": EfficientNetClassifier,
    "model_c": FoundationModelClassifier,
}


def train_fold(
    model_class,
    model_kwargs: dict,
    train_dataset: FrogDataset,
    val_dataset: FrogDataset,
    config: dict,
    fold: int,
    output_dir: Path,
):
    """Train a single fold."""
    pl.seed_everything(RANDOM_SEED + fold)

    model = model_class(**model_kwargs)

    train_labels = [m["label"] for m in train_dataset.metadata]
    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir / f"fold_{fold}",
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
    wandb_logger = WandbLogger(
        project="blue-frogs",
        name=f"{config.get('model_name', 'model')}_fold{fold}",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=wandb_logger,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    return checkpoint_cb.best_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--fold", type=int, default=None, help="Train single fold (for HPC)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model_name"] = args.model
    model_class = MODEL_CLASSES[args.model]
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and splits -- details depend on splits file format
    # (This is filled in during execution based on actual data)
    logger.info(f"Training {args.model} with config {args.config}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
