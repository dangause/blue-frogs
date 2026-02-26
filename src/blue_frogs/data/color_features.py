"""LAB color space feature extraction for axanthism detection.

The b* channel in LAB color space encodes the yellow-blue axis:
- Negative b* = blue
- Positive b* = yellow

This directly maps to the biology of axanthism: loss of yellow
xanthophore pigments causes the b* channel to shift negative.
"""

import cv2
import numpy as np


def extract_lab_features(image_rgb: np.ndarray) -> np.ndarray:
    """Extract LAB color space features from an RGB image.

    Returns a 30-dimensional feature vector:
    - 10 features per channel (L, a, b):
      mean, median, std, min, max, Q10, Q25, Q75, Q90, skewness
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    features = []
    for channel_idx in range(3):  # L, a, b
        ch = lab[:, :, channel_idx].ravel()
        features.extend([
            np.mean(ch),
            np.median(ch),
            np.std(ch),
            np.min(ch),
            np.max(ch),
            np.percentile(ch, 10),
            np.percentile(ch, 25),
            np.percentile(ch, 75),
            np.percentile(ch, 90),
            float(_skewness(ch)),
        ])

    return np.array(features, dtype=np.float32)


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of an array."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))
