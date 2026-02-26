"""Tests for PyTorch dataset and augmentation."""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from blue_frogs.data.dataset import (
    FrogDataset,
    get_train_transforms,
    get_val_transforms,
)


@pytest.fixture
def image_dir(tmp_path):
    """Create fake image directory structure."""
    for obs_id, color in [(123, (0, 128, 0)), (456, (50, 100, 200))]:
        obs_dir = tmp_path / str(obs_id)
        obs_dir.mkdir()
        img = Image.new("RGB", (200, 150), color=color)
        img.save(obs_dir / "001.jpg")
    return tmp_path


@pytest.fixture
def sample_metadata():
    return [
        {"observation_id": 123, "photo_id": 1, "photo_path": "123/001.jpg", "label": 0},
        {"observation_id": 456, "photo_id": 2, "photo_path": "456/001.jpg", "label": 1},
    ]


def test_frog_dataset_len(image_dir, sample_metadata):
    ds = FrogDataset(sample_metadata, image_dir, transform=get_val_transforms())
    assert len(ds) == 2


def test_frog_dataset_getitem_shape(image_dir, sample_metadata):
    ds = FrogDataset(sample_metadata, image_dir, transform=get_val_transforms())
    img, label = ds[0]
    assert img.shape == (3, 384, 384)
    assert isinstance(label, (int, float))


def test_train_transforms_preserve_shape(sample_image):
    transforms = get_train_transforms()
    img = np.array(Image.open(sample_image))
    result = transforms(image=img)
    assert result["image"].shape == (3, 384, 384)


def test_val_transforms_deterministic(sample_image):
    transforms = get_val_transforms()
    img = np.array(Image.open(sample_image))
    r1 = transforms(image=img)["image"]
    r2 = transforms(image=img)["image"]
    assert torch.allclose(r1, r2)
