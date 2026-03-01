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
from blue_frogs.models.model_b_classifier import FusionClassifier
from blue_frogs.models.model_c import FoundationModelClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "model_a": EfficientNetClassifier,
    "model_b": FusionClassifier,
    "model_c": FoundationModelClassifier,
}


def build_model_kwargs(model_name: str, config: dict) -> dict:
    """Extract model constructor kwargs from config.

    Each model has different config structure, so this maps config sections
    to the kwargs expected by each model class constructor.
    """
    training = config["training"]
    common_kwargs = {
        "loss_type": training["loss_type"],
        "pos_weight": training["pos_weight"],
        "optimizer": training["optimizer"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training.get("weight_decay", 0.0),
    }

    if model_name == "model_a":
        model_cfg = config["model"]
        return {
            **common_kwargs,
            "backbone": model_cfg["backbone"],
            "pretrained": model_cfg["pretrained"],
            "dropout": model_cfg["dropout"],
            "freeze_backbone": model_cfg.get("freeze_backbone", False),
        }
    elif model_name == "model_b":
        cls_cfg = config["classifier"]
        return {
            **common_kwargs,
            "backbone": cls_cfg["backbone"],
            "pretrained": cls_cfg["pretrained"],
            "color_feature_dim": cls_cfg["color_feature_dim"],
            "fusion_hidden": cls_cfg["fusion_hidden"],
            "dropout": cls_cfg["dropout"],
        }
    elif model_name == "model_c":
        model_cfg = config["model"]
        return {
            **common_kwargs,
            "backbone": model_cfg["backbone"],
            "model_size": model_cfg["model_size"],
            "hidden_dim": model_cfg["hidden_dim"],
            "dropout": model_cfg["dropout"],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


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


def load_training_data(
    labels_file: Path, splits_file: Path
) -> tuple[list[dict], list[dict]]:
    """Load metadata and split into train/test sets.

    labels_file: JSON list of {observation_id, photo_id, photo_path, label}
    splits_file: JSON with {train_indices: [...], test_indices: [...]}
    """
    import json

    with open(labels_file) as f:
        all_metadata = json.load(f)
    with open(splits_file) as f:
        splits = json.load(f)

    train_meta = [all_metadata[i] for i in splits["train_indices"]]
    test_meta = [all_metadata[i] for i in splits["test_indices"]]
    return train_meta, test_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--fold", type=int, default=None, help="Train single fold (for HPC)")
    parser.add_argument(
        "--precision", type=str, default="16-mixed",
        help="Training precision: 16-mixed, bf16-mixed, or 32",
    )
    parser.add_argument(
        "--wandb-offline", action="store_true",
        help="Run WandB in offline mode (sync logs later)",
    )
    args = parser.parse_args()

    if args.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model_name"] = args.model
    model_class = MODEL_CLASSES[args.model]
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_meta, test_meta = load_training_data(args.labels_file, args.splits_file)
    logger.info(f"Training {args.model} with config {args.config}")
    logger.info(f"Train: {len(train_meta)}, Test: {len(test_meta)}")
    logger.info(f"Output: {output_dir}")

    # Determine folds to train
    import pandas as pd
    train_df = pd.DataFrame(train_meta)
    n_folds = config.get("data", {}).get("n_folds", 5)
    seed = config.get("data", {}).get("seed", RANDOM_SEED)
    folds = get_fold_indices(train_df, n_folds=n_folds, seed=seed)

    if args.fold is not None:
        fold_list = [args.fold]
    else:
        fold_list = list(range(n_folds))

    image_dir = args.data_dir / "raw_images"
    model_kwargs = build_model_kwargs(args.model, config)
    results = {}

    for fold_idx in fold_list:
        logger.info(f"--- Fold {fold_idx}/{n_folds - 1} ---")
        train_idx, val_idx = folds[fold_idx]

        fold_train_meta = [train_meta[i] for i in train_idx]
        fold_val_meta = [train_meta[i] for i in val_idx]

        train_dataset = FrogDataset(fold_train_meta, image_dir, transform=get_train_transforms())
        val_dataset = FrogDataset(fold_val_meta, image_dir, transform=get_val_transforms())

        if args.model == "model_c" and "linear_probe" in config:
            best_path = train_fold_two_stage(
                model_class, model_kwargs, train_dataset, val_dataset,
                config, fold_idx, output_dir, args.precision,
            )
        else:
            best_path = train_fold(
                model_class, model_kwargs, train_dataset, val_dataset,
                config, fold_idx, output_dir, args.precision,
            )

        results[fold_idx] = {"best_checkpoint": best_path}
        logger.info(f"Fold {fold_idx} best checkpoint: {best_path}")

    # Save fold results summary
    import json
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
