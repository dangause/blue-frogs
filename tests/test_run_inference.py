"""Tests for multi-model inference script."""

import pytest


def test_get_model_class_returns_all_three():
    """get_model_class resolves all three model names."""
    from scripts.run_inference import get_model_class
    from blue_frogs.models.model_a import EfficientNetClassifier
    from blue_frogs.models.model_b_classifier import FusionClassifier
    from blue_frogs.models.model_c import FoundationModelClassifier

    assert get_model_class("model_a") is EfficientNetClassifier
    assert get_model_class("model_b") is FusionClassifier
    assert get_model_class("model_c") is FoundationModelClassifier


def test_get_model_class_raises_for_unknown():
    """get_model_class raises ValueError for unknown model."""
    from scripts.run_inference import get_model_class

    with pytest.raises(ValueError, match="Unknown model"):
        get_model_class("model_z")
