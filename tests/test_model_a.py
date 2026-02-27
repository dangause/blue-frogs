"""Tests for Model A: EfficientNetV2-S classifier."""

import pytest
import torch
from blue_frogs.models.model_a import EfficientNetClassifier


def test_efficientnet_forward_shape():
    model = EfficientNetClassifier(backbone="tf_efficientnetv2_s")
    x = torch.randn(2, 3, 384, 384)
    logits = model(x)
    assert logits.shape == (2, 1)


def test_efficientnet_training_step():
    model = EfficientNetClassifier(backbone="tf_efficientnetv2_s")
    x = torch.randn(2, 3, 384, 384)
    labels = torch.tensor([0, 1])
    loss = model.training_step((x, labels), 0)
    assert loss.item() > 0


def test_efficientnet_frozen_backbone():
    model = EfficientNetClassifier(backbone="tf_efficientnetv2_s", freeze_backbone=True)
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen"
