"""Tests for fold aggregation script."""

import json
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def mock_fold_results(tmp_path):
    """Create mock per-fold prediction files for aggregation testing."""
    results_dir = tmp_path / "model_a"
    n_samples = 50
    rng = np.random.RandomState(42)
    y_true = np.concatenate([np.ones(10), np.zeros(40)])

    for fold in range(3):  # 3 folds for speed
        fold_dir = results_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        # Simulate per-fold predictions
        y_score = rng.uniform(0, 1, size=n_samples)
        # Make positive samples score higher on average
        y_score[:10] += 0.3
        y_score = np.clip(y_score, 0, 1)
        preds = {
            "y_true": y_true.tolist(),
            "y_score": y_score.tolist(),
        }
        (fold_dir / "test_predictions.json").write_text(json.dumps(preds))

    return tmp_path


def test_aggregate_fold_predictions(mock_fold_results):
    """aggregate_fold_predictions averages predictions across folds."""
    from scripts.aggregate_folds import aggregate_fold_predictions

    y_true, y_score_ensemble = aggregate_fold_predictions(
        mock_fold_results / "model_a", n_folds=3
    )
    assert len(y_true) == 50
    assert len(y_score_ensemble) == 50
    assert 0.0 <= y_score_ensemble.min()
    assert y_score_ensemble.max() <= 1.0


def test_generate_comparison_table(mock_fold_results):
    """generate_comparison_table produces a markdown string."""
    from scripts.aggregate_folds import generate_comparison_table

    model_results = {
        "model_a": {
            "auprc": 0.85,
            "auprc_ci_lower": 0.80,
            "auprc_ci_upper": 0.90,
            "auroc": 0.92,
            "auroc_ci_lower": 0.88,
            "auroc_ci_upper": 0.96,
            "f1_optimal": 0.78,
        },
    }
    table = generate_comparison_table(model_results)
    assert "model_a" in table
    assert "AUPRC" in table
    assert "0.85" in table
