"""PyTorch Dataset and augmentation pipeline for frog images."""

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from blue_frogs.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> A.Compose:
    """Training augmentation pipeline.

    Hue augmentation is capped at +/- 5 degrees (out of 180 in OpenCV)
    because axanthism is a color-based trait.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.014,  # ~5 degrees / 360 = 0.014
            p=0.5,
        ),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Validation/test transforms (deterministic)."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class FrogDataset(Dataset):
    """Dataset for frog images with binary axanthism labels."""

    def __init__(
        self,
        metadata: list[dict[str, Any]],
        image_dir: Path,
        transform: A.Compose | None = None,
    ):
        self.metadata = metadata
        self.image_dir = Path(image_dir)
        self.transform = transform or get_val_transforms()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.metadata[idx]
        img_path = self.image_dir / entry["photo_path"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        label = entry["label"]

        return image_tensor, label
