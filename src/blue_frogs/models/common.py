"""Shared model components: losses, base Lightning module, samplers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler

from blue_frogs.evaluation.metrics import compute_classification_metrics


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class BaseClassifier(pl.LightningModule):
    """Base Lightning module for all frog classifiers."""

    def __init__(
        self,
        loss_type: str = "weighted_bce",
        pos_weight: float = 20.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "sgd",
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        if loss_type == "weighted_bce":
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Collect predictions for epoch-level metrics
        self._val_outputs: list[dict] = []

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits.squeeze(-1), labels.float())

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits.squeeze(-1))
        self._val_outputs.append({
            "loss": loss,
            "probs": probs.detach().cpu(),
            "labels": labels.detach().cpu(),
        })
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat([o["probs"] for o in self._val_outputs]).float().numpy()
        all_labels = torch.cat([o["labels"] for o in self._val_outputs]).float().numpy()
        avg_loss = torch.stack([o["loss"] for o in self._val_outputs]).mean()

        if len(set(all_labels)) >= 2:
            metrics = compute_classification_metrics(all_labels, all_probs)
            self.log("val/loss", avg_loss, prog_bar=True)
            self.log("val/auprc", metrics["auprc"], prog_bar=True)
            self.log("val/auroc", metrics["auroc"])
            self.log("val/f1", metrics["f1_optimal"])

        self._val_outputs.clear()

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create a weighted random sampler that oversamples the minority class."""
    import numpy as np
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    weights = 1.0 / class_counts[labels_arr]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(labels),
        replacement=True,
    )
