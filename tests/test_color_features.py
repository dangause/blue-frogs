"""Tests for LAB color feature extraction."""

import numpy as np
import pytest
from PIL import Image
from blue_frogs.data.color_features import extract_lab_features


def test_extract_lab_features_shape(sample_image):
    img = np.array(Image.open(sample_image))
    features = extract_lab_features(img)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    assert len(features) == 30  # 10 features per channel (L, a, b)


def test_extract_lab_features_blue_vs_green(sample_image, sample_blue_image):
    green_img = np.array(Image.open(sample_image))
    blue_img = np.array(Image.open(sample_blue_image))
    green_feats = extract_lab_features(green_img)
    blue_feats = extract_lab_features(blue_img)
    # b* channel (indices 20-29) should be more negative for blue images
    # In LAB, negative b* = blue, positive b* = yellow
    green_b_mean = green_feats[20]  # mean of b* channel
    blue_b_mean = blue_feats[20]
    assert blue_b_mean < green_b_mean
