"""Offline YOLO crop and LAB feature pre-computation for Model B.

Crops every image in the dataset through YOLO detection, saves crops to disk,
and pre-computes LAB color features from the cropped images. This ensures
train-time features match inference-time features (both from cropped frogs,
not full background images).

Usage:
    python scripts/preprocess_crops.py \
        --data-dir data/ \
        --labels-file data/labels.json \
        [--detector-checkpoint path/to/yolo.pt] \
        [--conf-threshold 0.5] \
        [--padding-fraction 0.1]
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from blue_frogs.data.color_features import extract_lab_features
from blue_frogs.models.model_b_detector import FrogDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def preprocess_crops(
    metadata: list[dict],
    raw_image_dir: Path,
    crop_dir: Path,
    conf_threshold: float = 0.5,
    padding_fraction: float = 0.1,
    detector_checkpoint: str | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run YOLO crop + LAB feature extraction for all images.

    Returns:
        lab_features: (N, 30) array of LAB features aligned with metadata order.
        index_map: {photo_path: row_index} mapping for lookup.
    """
    crop_dir.mkdir(parents=True, exist_ok=True)
    detector = FrogDetector(
        model_path=detector_checkpoint, conf_threshold=conf_threshold
    )

    lab_features = []
    index_map = {}
    stats = {"cropped": 0, "cached": 0, "failed": 0}

    for i, entry in enumerate(tqdm(metadata, desc="Cropping frogs")):
        photo_path = entry["photo_path"]
        crop_path = crop_dir / photo_path
        raw_path = raw_image_dir / photo_path

        # Try to load existing crop
        if crop_path.exists():
            image = cv2.imread(str(crop_path))
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                lab_features.append(extract_lab_features(image_rgb))
                index_map[photo_path] = i
                stats["cached"] += 1
                continue

        # Load raw image and crop
        if not raw_path.exists():
            lab_features.append(np.zeros(30, dtype=np.float32))
            index_map[photo_path] = i
            stats["failed"] += 1
            continue

        image = cv2.imread(str(raw_path))
        if image is None:
            lab_features.append(np.zeros(30, dtype=np.float32))
            index_map[photo_path] = i
            stats["failed"] += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = detector.crop_frog(image_rgb, padding_fraction=padding_fraction)

        if crop is not None and crop.size > 0:
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)
            lab_features.append(extract_lab_features(crop))
            stats["cropped"] += 1
        else:
            lab_features.append(np.zeros(30, dtype=np.float32))
            stats["failed"] += 1

        index_map[photo_path] = i

    logger.info(
        "Crop stats: %d cropped, %d cached, %d failed",
        stats["cropped"], stats["cached"], stats["failed"],
    )
    return np.stack(lab_features).astype(np.float32), index_map


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute YOLO crops and LAB features for Model B training"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=str, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--padding-fraction", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.labels_file) as f:
        metadata = json.load(f)

    raw_image_dir = args.data_dir / "raw_images"
    crop_dir = args.data_dir / "cropped_images"

    logger.info("Processing %d images from %s", len(metadata), raw_image_dir)

    lab_features, index_map = preprocess_crops(
        metadata=metadata,
        raw_image_dir=raw_image_dir,
        crop_dir=crop_dir,
        conf_threshold=args.conf_threshold,
        padding_fraction=args.padding_fraction,
        detector_checkpoint=args.detector_checkpoint,
    )

    # Save LAB features and index
    lab_path = args.data_dir / "crop_lab_features.npy"
    index_path = args.data_dir / "crop_lab_index.json"

    np.save(lab_path, lab_features)
    with open(index_path, "w") as f:
        json.dump(index_map, f)

    logger.info("Saved LAB features (%s) to %s", lab_features.shape, lab_path)
    logger.info("Saved index (%d entries) to %s", len(index_map), index_path)


if __name__ == "__main__":
    main()
