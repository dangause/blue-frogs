"""Model D: DINOv3 classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from blue_frogs.models.common import BaseClassifier

_DINOV3_PATCH_SIZE = 16

_DINOV3_MODEL_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class DINOv3Classifier(BaseClassifier):
    """DINOv3 vision transformer with classification head."""

    def __init__(
        self,
        backbone: str = "dinov3",
        model_size: str = "large",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone, feature_dim = self._load_dinov3(model_size, pretrained)

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

    def _load_dinov3(self, model_size: str, pretrained: bool):
        """Load DINOv3 from HuggingFace Transformers."""
        model_id = _DINOV3_MODEL_IDS.get(model_size)
        if model_id is None:
            raise ValueError(
                f"Unknown model_size: {model_size}. "
                f"Choose from: {list(_DINOV3_MODEL_IDS.keys())}"
            )

        if pretrained:
            model = AutoModel.from_pretrained(model_id)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            model = AutoModel.from_config(config)

        feature_dim = model.config.hidden_size
        return model, feature_dim

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
        x = self._pad_to_patch_multiple(x, _DINOV3_PATCH_SIZE)
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        return self.head(features)

    def configure_optimizers(self):
        """Differential learning rates: low for backbone, high for head."""
        if self.hparams.freeze_backbone:
            return super().configure_optimizers()

        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.learning_rate * 0.01},
                {"params": head_params, "lr": self.learning_rate},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
