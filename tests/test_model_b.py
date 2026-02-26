"""Tests for Model B: two-stage detection + classification."""

import pytest
import torch
from blue_frogs.models.model_b_classifier import FusionClassifier


def test_fusion_classifier_forward():
    model = FusionClassifier(backbone="tf_efficientnetv2_s", color_feature_dim=30)
    images = torch.randn(2, 3, 384, 384)
    color_feats = torch.randn(2, 30)
    logits = model(images, color_feats)
    assert logits.shape == (2, 1)
