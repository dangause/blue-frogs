"""Tests for Model C: foundation model classifiers."""

import pytest
import torch
from blue_frogs.models.model_c import FoundationModelClassifier


def test_dinov2_forward_shape():
    model = FoundationModelClassifier(backbone="dinov2", model_size="small")
    x = torch.randn(2, 3, 384, 384)
    logits = model(x)
    assert logits.shape == (2, 1)


def test_linear_probe_freezes_backbone():
    model = FoundationModelClassifier(
        backbone="dinov2", model_size="small", freeze_backbone=True
    )
    backbone_params = list(model.backbone.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    head_params = list(model.head.parameters())
    assert all(p.requires_grad for p in head_params)
