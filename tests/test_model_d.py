"""Tests for Model D: DINOv3 classifier."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from blue_frogs.models.model_d import DINOv3Classifier


@pytest.fixture
def mock_dinov3_model():
    """Create a mock HuggingFace DINOv3 model."""
    mock_model = MagicMock()
    mock_model.config.hidden_size = 1024
    # Simulate forward pass returning BaseModelOutput with last_hidden_state
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 577, 1024)  # (batch, seq_len, hidden)
    mock_model.return_value = mock_output
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(2, 2))])
    mock_model.named_parameters.return_value = iter(
        [("weight", torch.nn.Parameter(torch.randn(2, 2)))]
    )
    return mock_model


@patch("blue_frogs.models.model_d.AutoModel")
def test_dinov3_forward_shape(mock_auto_model, mock_dinov3_model):
    """Forward pass produces (batch_size, 1) logits."""
    mock_auto_model.from_pretrained.return_value = mock_dinov3_model

    model = DINOv3Classifier(backbone="dinov3", model_size="large")
    x = torch.randn(2, 3, 384, 384)
    logits = model(x)
    assert logits.shape == (2, 1)


@patch("blue_frogs.models.model_d.AutoModel")
def test_dinov3_freeze_backbone(mock_auto_model, mock_dinov3_model):
    """freeze_backbone=True freezes backbone params but not head."""
    mock_auto_model.from_pretrained.return_value = mock_dinov3_model

    model = DINOv3Classifier(
        backbone="dinov3", model_size="large", freeze_backbone=True
    )
    backbone_params = list(model.backbone.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    head_params = list(model.head.parameters())
    assert all(p.requires_grad for p in head_params)


@patch("blue_frogs.models.model_d.AutoModel")
def test_dinov3_head_dimensions(mock_auto_model, mock_dinov3_model):
    """Head uses correct input dim from backbone config."""
    mock_auto_model.from_pretrained.return_value = mock_dinov3_model

    model = DINOv3Classifier(
        backbone="dinov3", model_size="large", hidden_dim=256
    )
    # First layer of head is LayerNorm with backbone's hidden_size
    layer_norm = model.head[0]
    assert layer_norm.normalized_shape == (1024,)
    # Second layer maps hidden_size -> hidden_dim
    linear1 = model.head[1]
    assert linear1.in_features == 1024
    assert linear1.out_features == 256


@patch("blue_frogs.models.model_d.AutoModel")
def test_dinov3_model_size_mapping(mock_auto_model, mock_dinov3_model):
    """Each model_size maps to the correct HuggingFace model ID."""
    mock_auto_model.from_pretrained.return_value = mock_dinov3_model

    for size, expected_suffix in [
        ("small", "dinov3-vits16-pretrain-lvd1689m"),
        ("base", "dinov3-vitb16-pretrain-lvd1689m"),
        ("large", "dinov3-vitl16-pretrain-lvd1689m"),
    ]:
        mock_auto_model.from_pretrained.reset_mock()
        DINOv3Classifier(backbone="dinov3", model_size=size)
        mock_auto_model.from_pretrained.assert_called_once()
        call_args = mock_auto_model.from_pretrained.call_args
        assert expected_suffix in call_args[0][0]


def test_pad_to_patch_multiple():
    """Input is padded to multiple of 16."""
    # 383 is not a multiple of 16; should pad to 384
    x = torch.randn(1, 3, 383, 383)
    padded = DINOv3Classifier._pad_to_patch_multiple(x, 16)
    assert padded.shape[2] % 16 == 0
    assert padded.shape[3] % 16 == 0
