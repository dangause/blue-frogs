"""Tests for model comparison statistics."""

import numpy as np
from blue_frogs.evaluation.comparison import mcnemar_test, compare_models


def test_mcnemar_test_identical_models():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    preds_a = np.array([0, 0, 1, 1, 0, 1])
    preds_b = np.array([0, 0, 1, 1, 0, 1])
    result = mcnemar_test(y_true, preds_a, preds_b)
    assert result["p_value"] >= 0.05  # no significant difference


def test_compare_models_returns_table():
    y_true = np.array([0] * 50 + [1] * 10)
    rng = np.random.RandomState(42)
    scores = {
        "model_a": rng.random(60),
        "model_b": rng.random(60),
    }
    table = compare_models(y_true, scores)
    assert "model_a" in table
    assert "model_b" in table
    assert "auprc" in table["model_a"]
