"""Tests for the unified training script."""

import inspect
import json

import numpy as np
import pytest
import yaml
from pathlib import Path
from PIL import Image


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


@pytest.fixture
def synthetic_training_data(tmp_path):
    """Create a minimal dataset with labels, images, and splits for testing main()."""
    image_dir = tmp_path / "raw_images"
    image_dir.mkdir()
    metadata = []
    n_positive = 10
    n_negative = 40

    for i in range(n_positive + n_negative):
        label = 1 if i < n_positive else 0
        obs_id = 1000 + i
        photo_id = 2000 + i
        obs_dir = image_dir / str(obs_id)
        obs_dir.mkdir(exist_ok=True)
        img = Image.new("RGB", (100, 100), color=(50, 100, 200) if label else (0, 128, 0))
        photo_path = f"{obs_id}/{photo_id}.jpg"
        img.save(image_dir / photo_path)
        metadata.append({
            "observation_id": obs_id,
            "photo_id": photo_id,
            "photo_path": photo_path,
            "label": label,
        })

    labels_file = tmp_path / "labels.json"
    labels_file.write_text(json.dumps(metadata))

    rng = np.random.RandomState(42)
    indices = list(range(len(metadata)))
    rng.shuffle(indices)
    test_size = int(0.2 * len(metadata))
    splits = {
        "test_indices": indices[:test_size],
        "train_indices": indices[test_size:],
    }
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps(splits))

    return tmp_path, labels_file, splits_file, image_dir


def test_load_training_data(synthetic_training_data):
    """load_training_data returns train_meta and test_meta with expected columns."""
    from scripts.train import load_training_data

    data_dir, labels_file, splits_file, image_dir = synthetic_training_data
    train_meta, test_meta = load_training_data(labels_file, splits_file)

    assert len(train_meta) > 0
    assert len(test_meta) > 0
    assert len(train_meta) + len(test_meta) == 50
    assert all("label" in m for m in train_meta)
    assert all("photo_path" in m for m in train_meta)


def test_train_fold_accepts_precision():
    """train_fold accepts a precision parameter."""
    from scripts.train import train_fold
    sig = inspect.signature(train_fold)
    assert "precision" in sig.parameters


def test_train_fold_two_stage_exists():
    """train_fold_two_stage function exists and has expected parameters."""
    from scripts.train import train_fold_two_stage
    sig = inspect.signature(train_fold_two_stage)
    assert "precision" in sig.parameters
    assert "config" in sig.parameters


def test_save_test_predictions_creates_file(tmp_path):
    """save_test_predictions writes y_true and y_score to JSON."""
    from scripts.train import save_test_predictions

    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]
    save_test_predictions(y_true, y_score, tmp_path)

    pred_file = tmp_path / "test_predictions.json"
    assert pred_file.exists()
    data = json.loads(pred_file.read_text())
    assert data["y_true"] == y_true
    assert data["y_score"] == y_score
