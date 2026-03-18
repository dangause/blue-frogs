"""Unified training script for all model architectures."""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from blue_frogs.config import MODEL_DIR, RANDOM_SEED
from blue_frogs.data.dataset import FrogColorDataset, FrogDataset, get_train_transforms, get_val_transforms
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
            "freeze_backbone": cls_cfg.get("freeze_backbone", False),
        }
    elif model_name == "model_c":
        model_cfg = config["model"]
        return {
            **common_kwargs,
            "backbone": model_cfg["backbone"],
            "pretrained": model_cfg.get("pretrained", True),
            "model_size": model_cfg["model_size"],
            "hidden_dim": model_cfg["hidden_dim"],
            "dropout": model_cfg["dropout"],
            "freeze_backbone": model_cfg.get("freeze_backbone", False),
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
    precision: str = "16-mixed",
):
    """Train a single fold."""
    pl.seed_everything(RANDOM_SEED + fold)

    model = model_class(**model_kwargs)

    train_labels = [m["label"] for m in train_dataset.metadata]
    sampler = make_weighted_sampler(train_labels)

    num_workers = config["training"].get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=fold_dir,
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
        precision=precision,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=wandb_logger,
        deterministic="warn",
    )
    trainer.fit(model, train_loader, val_loader)

    return checkpoint_cb.best_model_path


def train_fold_two_stage(
    model_class,
    model_kwargs: dict,
    train_dataset: FrogDataset,
    val_dataset: FrogDataset,
    config: dict,
    fold: int,
    output_dir: Path,
    precision: str = "16-mixed",
):
    """Train Model C in two stages: linear probe (frozen) then fine-tune (unfrozen)."""
    pl.seed_everything(RANDOM_SEED + fold)

    num_workers = config["training"].get("num_workers", 4)
    batch_size = config["training"]["batch_size"]
    fold_dir = output_dir / f"fold_{fold}"

    # --- Stage 1: Linear probe (frozen backbone) ---
    logger.info(f"Fold {fold} — Stage 1: Linear Probe (frozen backbone)")
    probe_cfg = config["linear_probe"]

    probe_kwargs = {
        **model_kwargs,
        "freeze_backbone": True,
        "learning_rate": probe_cfg["learning_rate"],
    }
    model = model_class(**probe_kwargs)

    train_labels = [m["label"] for m in train_dataset.metadata]
    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    probe_dir = fold_dir / "linear_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=probe_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )
    wandb_logger = WandbLogger(
        project="blue-frogs",
        name=f"{config.get('model_name', 'model')}_fold{fold}_probe",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=probe_cfg["max_epochs"], accelerator="auto", precision=precision,
        callbacks=[checkpoint_cb], logger=wandb_logger, deterministic="warn",
    )
    trainer.fit(model, train_loader, val_loader)
    probe_ckpt = checkpoint_cb.best_model_path
    logger.info(f"Linear probe complete. Best: {probe_ckpt}")

    # --- Stage 2: Fine-tune (unfrozen backbone, differential LR) ---
    logger.info(f"Fold {fold} — Stage 2: Fine-tune (unfrozen backbone)")
    ft_cfg = config["finetune"]

    probe_model = model_class.load_from_checkpoint(probe_ckpt)
    ft_kwargs = {**model_kwargs, "freeze_backbone": False}
    model = model_class(**ft_kwargs)
    model.head.load_state_dict(probe_model.head.state_dict())

    ft_batch_size = ft_cfg.get("batch_size", batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=ft_batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=ft_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    ft_dir = fold_dir / "finetune"
    ft_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ft_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc", mode="max",
        patience=config["training"]["early_stopping_patience"],
    )
    wandb_logger = WandbLogger(
        project="blue-frogs",
        name=f"{config.get('model_name', 'model')}_fold{fold}_finetune",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=ft_cfg.get("max_epochs", config["training"]["max_epochs"]),
        accelerator="auto", precision=precision,
        callbacks=[checkpoint_cb, early_stop_cb], logger=wandb_logger,
        deterministic="warn",
    )
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Fine-tune complete. Best: {checkpoint_cb.best_model_path}")

    return checkpoint_cb.best_model_path


def load_training_data(
    labels_file: Path, splits_file: Path
) -> tuple[list[dict], list[dict]]:
    """Load metadata and split into train/test sets.

    labels_file: JSON list of {observation_id, photo_id, photo_path, label}
    splits_file: JSON with {train_indices: [...], test_indices: [...]}
    """
    with open(labels_file) as f:
        all_metadata = json.load(f)
    with open(splits_file) as f:
        splits = json.load(f)

    train_meta = [all_metadata[i] for i in splits["train_indices"]]
    test_meta = [all_metadata[i] for i in splits["test_indices"]]
    return train_meta, test_meta


def save_test_predictions(
    y_true: list, y_score: list, fold_dir: Path, logits: list | None = None
) -> None:
    """Save test-set predictions for later aggregation and calibration."""
    data = {"y_true": y_true, "y_score": y_score}
    if logits is not None:
        data["logits"] = logits
    pred_path = fold_dir / "test_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(data, f)


def save_training_metadata(
    fold_dir: Path,
    model_name: str,
    fold: int,
    config: dict,
    train_meta: list[dict],
    val_meta: list[dict],
    best_checkpoint: str,
) -> None:
    """Save training metadata for reproducibility and model cards."""
    # Git info
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git_dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit, git_branch, git_dirty = "unknown", "unknown", False

    # Data stats
    train_labels = [m["label"] for m in train_meta]
    val_labels = [m["label"] for m in val_meta]

    # Compute the actual seed used for this fold
    fold_seed = RANDOM_SEED + fold

    metadata = {
        "model_name": model_name,
        "fold": fold,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "dirty": git_dirty,
        },
        "random_seeds": {
            "base_seed": RANDOM_SEED,
            "fold_seed": fold_seed,
            "numpy_seed": fold_seed,
            "torch_seed": fold_seed,
            "cuda_seed": fold_seed if torch.cuda.is_available() else None,
        },
        "config": config,
        "data_stats": {
            "n_train": len(train_meta),
            "n_val": len(val_meta),
            "pos_ratio_train": sum(train_labels) / len(train_labels) if train_labels else 0,
            "pos_ratio_val": sum(val_labels) / len(val_labels) if val_labels else 0,
            "n_pos_train": sum(train_labels),
            "n_pos_val": sum(val_labels),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "pytorch_lightning_version": pl.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "best_checkpoint": str(best_checkpoint),
    }

    meta_path = fold_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Training metadata saved to {meta_path}")


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
    # Model B crop / precomputed LAB args
    parser.add_argument(
        "--crop-dir", type=Path, default=None,
        help="Directory with YOLO-cropped images (Model B)",
    )
    parser.add_argument(
        "--lab-features", type=Path, default=None,
        help="Path to pre-computed LAB features .npy file (Model B)",
    )
    parser.add_argument(
        "--lab-index", type=Path, default=None,
        help="Path to LAB index .json mapping photo_path -> row index (Model B)",
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

    # Load precomputed LAB features for Model B if provided
    precomputed_lab = None
    if args.model == "model_b" and args.lab_features and args.lab_index:
        import numpy as np
        lab_array = np.load(args.lab_features)
        with open(args.lab_index) as f:
            lab_index = json.load(f)
        # Build lookup: photo_path -> feature vector
        precomputed_lab = {
            photo_path: lab_array[int(row_idx)]
            for photo_path, row_idx in lab_index.items()
        }
        logger.info("Loaded precomputed LAB features: %d entries", len(precomputed_lab))

    for fold_idx in fold_list:
        logger.info(f"--- Fold {fold_idx}/{n_folds - 1} ---")
        train_idx, val_idx = folds[fold_idx]

        fold_train_meta = [train_meta[i] for i in train_idx]
        fold_val_meta = [train_meta[i] for i in val_idx]

        if args.model == "model_b":
            train_dataset = FrogColorDataset(
                fold_train_meta, image_dir, transform=get_train_transforms(),
                crop_dir=args.crop_dir, precomputed_lab=precomputed_lab,
            )
            val_dataset = FrogColorDataset(
                fold_val_meta, image_dir, transform=get_val_transforms(),
                crop_dir=args.crop_dir, precomputed_lab=precomputed_lab,
            )
        else:
            train_dataset = FrogDataset(fold_train_meta, image_dir, transform=get_train_transforms())
            val_dataset = FrogDataset(fold_val_meta, image_dir, transform=get_val_transforms())

        use_two_stage = "linear_probe" in config
        if use_two_stage:
            best_path = train_fold_two_stage(
                model_class, model_kwargs, train_dataset, val_dataset,
                config, fold_idx, output_dir, args.precision,
            )
        else:
            best_path = train_fold(
                model_class, model_kwargs, train_dataset, val_dataset,
                config, fold_idx, output_dir, args.precision,
            )

        # Evaluate on held-out test set and save predictions
        if args.model == "model_b":
            test_dataset = FrogColorDataset(
                test_meta, image_dir, transform=get_val_transforms(),
                crop_dir=args.crop_dir, precomputed_lab=precomputed_lab,
            )
        else:
            test_dataset = FrogDataset(test_meta, image_dir, transform=get_val_transforms())
        test_loader = DataLoader(
            test_dataset, batch_size=config["training"]["batch_size"],
            shuffle=False, num_workers=config["training"].get("num_workers", 4),
        )

        loaded_model = model_class.load_from_checkpoint(best_path)
        loaded_model.eval()
        loaded_model.float()  # ensure float32 for MPS/CPU test evaluation
        loaded_model.cpu()    # avoid MPS tensor type mismatches
        all_probs, all_labels, all_logits = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                if args.model == "model_b":
                    images, color_feats, labels = batch
                    logits = loaded_model(images.float(), color_feats.float())
                else:
                    images, labels = batch
                    logits = loaded_model(images.float())
                logits_squeezed = logits.squeeze(-1)
                probs = torch.sigmoid(logits_squeezed)
                all_probs.extend(probs.cpu().tolist())
                all_logits.extend(logits_squeezed.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_test_predictions(all_labels, all_probs, fold_dir, logits=all_logits)

        # Save training metadata for reproducibility
        save_training_metadata(
            fold_dir=fold_dir,
            model_name=args.model,
            fold=fold_idx,
            config=config,
            train_meta=fold_train_meta,
            val_meta=fold_val_meta,
            best_checkpoint=best_path,
        )

        results[fold_idx] = {"best_checkpoint": best_path}
        logger.info(f"Fold {fold_idx} best checkpoint: {best_path}")

    # Save fold results summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
