"""Tests for evaluation metrics."""

import pytest
import numpy as np
from blue_frogs.evaluation.metrics import (
    compute_classification_metrics,
    compute_bootstrap_ci,
    find_threshold_at_recall,
    find_threshold_at_precision,
)


def test_compute_classification_metrics_perfect():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    metrics = compute_classification_metrics(y_true, y_score)
    assert metrics["auroc"] > 0.99
    assert metrics["auprc"] > 0.99


def test_compute_classification_metrics_random():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 100)
    y_score = rng.random(100)
    metrics = compute_classification_metrics(y_true, y_score)
    assert 0.0 < metrics["auroc"] < 1.0
    assert "f1_optimal" in metrics
    assert "optimal_threshold" in metrics


def test_compute_bootstrap_ci():
    rng = np.random.RandomState(42)
    y_true = np.array([0] * 90 + [1] * 10)
    # Overlapping scores so AUROC varies across bootstrap samples
    y_score = np.concatenate([rng.random(90) * 0.8, 0.2 + rng.random(10) * 0.8])
    ci = compute_bootstrap_ci(y_true, y_score, metric="auroc", n_bootstrap=100, seed=42)
    assert ci["lower"] < ci["mean"] < ci["upper"]


def test_find_threshold_at_recall():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    threshold = find_threshold_at_recall(y_true, y_score, target_recall=0.95)
    assert 0.0 <= threshold <= 1.0


def test_find_threshold_at_precision():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    threshold = find_threshold_at_precision(y_true, y_score, target_precision=0.95)
    assert 0.0 <= threshold <= 1.0
