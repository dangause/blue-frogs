"""Statistical comparison of model performance."""

import numpy as np
from scipy.stats import chi2

from blue_frogs.evaluation.metrics import (
    compute_classification_metrics,
    compute_bootstrap_ci,
)


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """McNemar's test comparing two classifiers' predictions."""
    # Count discordant pairs
    a_correct_b_wrong = np.sum((preds_a == y_true) & (preds_b != y_true))
    a_wrong_b_correct = np.sum((preds_a != y_true) & (preds_b == y_true))

    n = a_correct_b_wrong + a_wrong_b_correct
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0}

    # McNemar's test with continuity correction
    statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / n
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_discordant": int(n),
        "a_correct_b_wrong": int(a_correct_b_wrong),
        "a_wrong_b_correct": int(a_wrong_b_correct),
    }


def compare_models(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compare multiple models on the same test set.

    Returns a dict of model_name -> metrics (with bootstrap CIs).
    """
    results = {}
    for name, scores in model_scores.items():
        metrics = compute_classification_metrics(y_true, scores)
        for metric_name in ["auroc", "auprc"]:
            ci = compute_bootstrap_ci(y_true, scores, metric_name, n_bootstrap, seed=seed)
            metrics[f"{metric_name}_ci_lower"] = ci["lower"]
            metrics[f"{metric_name}_ci_upper"] = ci["upper"]
        results[name] = metrics
    return results
