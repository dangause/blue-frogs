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
from blue_frogs.data.color_features import extract_lab_features

logger = __import__("logging").getLogger(__name__)


def _safe_load_image(path: Path) -> np.ndarray:
    """Load an image, returning a black placeholder if the file is missing or corrupt."""
    image = cv2.imread(str(path))
    if image is None:
        logger.warning("Failed to load image: %s — using placeholder", path)
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
        image = _safe_load_image(img_path)

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        label = entry["label"]

        return image_tensor, label


class FrogColorDataset(Dataset):
    """Dataset for Model B: returns (image, color_features, label).

    Supports two modes:
    1. Precomputed: pass crop_dir and precomputed_lab to use YOLO-cropped
       images and pre-extracted LAB features (matches inference pipeline).
    2. On-the-fly: extracts LAB from loaded image at runtime (legacy).
    """

    def __init__(
        self,
        metadata: list[dict[str, Any]],
        image_dir: Path,
        transform: A.Compose | None = None,
        crop_dir: Path | None = None,
        precomputed_lab: dict[str, np.ndarray] | None = None,
    ):
        self.metadata = metadata
        self.image_dir = Path(image_dir)
        self.transform = transform or get_val_transforms()
        self.crop_dir = Path(crop_dir) if crop_dir is not None else None
        self.precomputed_lab = precomputed_lab

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.metadata[idx]
        photo_path = entry["photo_path"]

        # Prefer crop, fall back to raw
        if self.crop_dir is not None:
            crop_path = self.crop_dir / photo_path
            img_path = crop_path if crop_path.exists() else self.image_dir / photo_path
        else:
            img_path = self.image_dir / photo_path

        image = _safe_load_image(img_path)

        # Use precomputed LAB or extract on-the-fly
        if self.precomputed_lab is not None and photo_path in self.precomputed_lab:
            color_feats = self.precomputed_lab[photo_path]
            color_tensor = torch.from_numpy(color_feats)
        else:
            color_feats = extract_lab_features(image)
            color_tensor = torch.from_numpy(color_feats)

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        label = entry["label"]

        return image_tensor, color_tensor, label
