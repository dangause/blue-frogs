"""Evaluation metrics for binary classification with class imbalance."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Compute all classification metrics for a single evaluation."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # F1 at each threshold
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_score >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1_optimal": float(f1_scores[best_idx]),
        "optimal_threshold": float(optimal_threshold),
        "precision_at_optimal": float(precision[best_idx]),
        "recall_at_optimal": float(recall[best_idx]),
        "confusion_matrix": cm.tolist(),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auroc",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence intervals for a metric."""
    rng = np.random.RandomState(seed)
    metric_fn = {"auroc": roc_auc_score, "auprc": average_precision_score}[metric]

    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))

    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(scores)),
        "lower": float(np.percentile(scores, 100 * alpha)),
        "upper": float(np.percentile(scores, 100 * (1 - alpha))),
    }


def find_threshold_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.95
) -> float:
    """Find the decision threshold that achieves at least target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = recall >= target_recall
    if not valid.any():
        return 0.0
    # Among thresholds achieving target recall, pick highest (most conservative)
    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(precision[valid_idx])]
    return float(thresholds[best]) if best < len(thresholds) else 0.0


def find_threshold_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.95
) -> float:
    """Find the decision threshold that achieves at least target_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = precision >= target_precision
    if not valid.any():
        return 1.0
    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(recall[valid_idx])]
    return float(thresholds[best]) if best < len(thresholds) else 1.0
