"""Model B Stage 2: CNN + LAB color feature fusion classifier."""

import timm
import torch
import torch.nn as nn

from blue_frogs.models.common import BaseClassifier


class FusionClassifier(BaseClassifier):
    """EfficientNetV2-S backbone fused with explicit LAB color features."""

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
        color_feature_dim: int = 30,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        cnn_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(cnn_dim + color_feature_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self, images: torch.Tensor, color_features: torch.Tensor
    ) -> torch.Tensor:
        cnn_features = self.backbone(images)
        fused = torch.cat([cnn_features, color_features], dim=-1)
        return self.head(fused)

    def training_step(self, batch, batch_idx):
        images, color_feats, labels = batch
        logits = self(images, color_feats)
        loss = self.compute_loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, color_feats, labels = batch
        logits = self(images, color_feats)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits.squeeze(-1))
        self._val_outputs.append({
            "loss": loss,
            "probs": probs.detach().cpu(),
            "labels": labels.detach().cpu(),
        })
        return loss
