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

    ft_batch_size = ft_cfg.get("batch_size", batch_size)
    train_loader, val_loader = _make_loaders(train_meta, val_meta, image_dir, ft_batch_size, num_workers)

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
