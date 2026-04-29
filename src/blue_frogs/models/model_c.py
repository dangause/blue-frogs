"""Model C: Foundation model classifiers (DINOv2 / BioCLIP)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from blue_frogs.models.common import BaseClassifier

# DINOv2 uses 14px patches; input must be a multiple of 14
_DINOV2_PATCH_SIZE = 14


class FoundationModelClassifier(BaseClassifier):
    """Vision foundation model with classification head."""

    def __init__(
        self,
        backbone: str = "dinov2",
        model_size: str = "base",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.backbone_name = backbone

        if backbone == "dinov2":
            self.backbone, feature_dim = self._load_dinov2(model_size)
        elif backbone == "bioclip":
            self.backbone, feature_dim = self._load_bioclip()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _load_dinov2(self, model_size: str):
        """Load DINOv2 from torch hub."""
        model_names = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
        }
        model_name = model_names.get(model_size, "dinov2_vitb14")
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        feature_dim = model.embed_dim
        return model, feature_dim

    def _load_bioclip(self):
        """Load BioCLIP vision encoder."""
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )
        # Extract only the visual encoder
        visual = model.visual
        feature_dim = visual.output_dim
        return visual, feature_dim

    @staticmethod
    def _pad_to_patch_multiple(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Pad input so spatial dims are multiples of patch_size."""
        _, _, h, w = x.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "dinov2":
            x = self._pad_to_patch_multiple(x, _DINOV2_PATCH_SIZE)
            features = self.backbone(x)  # CLS token
        elif self.backbone_name == "bioclip":
            features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)

    def configure_optimizers(self):
        """Differential learning rates: low for backbone, high for head."""
        if self.hparams.freeze_backbone:
            return super().configure_optimizers()

        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.learning_rate * 0.01},
            {"params": head_params, "lr": self.learning_rate},
        ], weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
