"""Model A: EfficientNetV2-S ensemble classifier."""

import timm
import torch
import torch.nn as nn

from blue_frogs.models.common import BaseClassifier


class EfficientNetClassifier(BaseClassifier):
    """EfficientNetV2-S with binary classification head."""

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class EfficientNetEnsemble:
    """Ensemble of EfficientNetV2-S models from k-fold training."""

    def __init__(self, checkpoint_paths: list[str]):
        self.models = []
        for path in checkpoint_paths:
            model = EfficientNetClassifier.load_from_checkpoint(path)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all ensemble members."""
        logits = torch.stack([m(x) for m in self.models])
        avg_logits = logits.mean(dim=0)
        return torch.sigmoid(avg_logits.squeeze(-1))
