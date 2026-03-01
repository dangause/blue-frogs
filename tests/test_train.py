"""Tests for the unified training script."""

import pytest
import yaml


def test_build_model_kwargs_model_a():
    """build_model_kwargs extracts the right constructor args for Model A."""
    from scripts.train import build_model_kwargs

    config = {
        "model": {
            "backbone": "tf_efficientnetv2_s",
            "pretrained": True,
            "dropout": 0.3,
            "freeze_backbone": False,
        },
        "training": {
            "loss_type": "weighted_bce",
            "pos_weight": 20.0,
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
    }
    kwargs = build_model_kwargs("model_a", config)
    assert kwargs["backbone"] == "tf_efficientnetv2_s"
    assert kwargs["pretrained"] is True
    assert kwargs["dropout"] == 0.3
    assert kwargs["loss_type"] == "weighted_bce"
    assert kwargs["pos_weight"] == 20.0
    assert kwargs["optimizer"] == "sgd"
    assert kwargs["learning_rate"] == 0.001


def test_build_model_kwargs_model_b():
    """build_model_kwargs extracts fusion-specific args for Model B."""
    from scripts.train import build_model_kwargs

    config = {
        "classifier": {
            "backbone": "tf_efficientnetv2_s",
            "pretrained": True,
            "color_feature_dim": 30,
            "fusion_hidden": 256,
            "dropout": 0.3,
        },
        "training": {
            "loss_type": "weighted_bce",
            "pos_weight": 20.0,
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
    }
    kwargs = build_model_kwargs("model_b", config)
    assert kwargs["color_feature_dim"] == 30
    assert kwargs["fusion_hidden"] == 256
    assert kwargs["backbone"] == "tf_efficientnetv2_s"


def test_build_model_kwargs_model_c():
    """build_model_kwargs extracts foundation model args for Model C."""
    from scripts.train import build_model_kwargs

    config = {
        "model": {
            "backbone": "dinov2",
            "model_size": "base",
            "hidden_dim": 256,
            "dropout": 0.3,
        },
        "training": {
            "loss_type": "weighted_bce",
            "pos_weight": 20.0,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
        },
    }
    kwargs = build_model_kwargs("model_c", config)
    assert kwargs["backbone"] == "dinov2"
    assert kwargs["model_size"] == "base"
    assert kwargs["hidden_dim"] == 256
    assert kwargs["optimizer"] == "adamw"


def test_model_classes_contains_all_three():
    """MODEL_CLASSES dict has entries for all three models."""
    from scripts.train import MODEL_CLASSES

    assert "model_a" in MODEL_CLASSES
    assert "model_b" in MODEL_CLASSES
    assert "model_c" in MODEL_CLASSES
