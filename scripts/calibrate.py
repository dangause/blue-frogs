"""Post-hoc temperature scaling and threshold optimization.

Loads test predictions from each model's fold dirs, fits a temperature
parameter via NLL minimization, then finds recall-targeted thresholds.
Saves calibration params (T, thresholds) to results/calibration.json.

Usage:
    python scripts/calibrate.py --results-dir results/ --target-recall 0.95
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve

from blue_frogs.evaluation.metrics import (
    compute_classification_metrics,
    find_threshold_at_recall,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_fold_predictions(model_dir: Path) -> dict:
    """Load and merge test predictions across folds for a single model.

    Returns dict with 'y_true' and 'y_score' arrays, plus 'logits' if available.
    """
    all_true, all_score, all_logits = [], [], []
    has_logits = True

    for fold_dir in sorted(model_dir.glob("fold_*")):
        pred_file = fold_dir / "test_predictions.json"
        if not pred_file.exists():
            # Check in finetune subdir for two-stage models
            pred_file = fold_dir / "finetune" / "test_predictions.json"
        if not pred_file.exists():
            logger.warning("No test_predictions.json in %s, skipping", fold_dir)
            continue

        with open(pred_file) as f:
            preds = json.load(f)

        all_true.extend(preds["y_true"])
        all_score.extend(preds["y_score"])

        if "logits" in preds:
            all_logits.extend(preds["logits"])
        else:
            has_logits = False

    if not all_true:
        return {}

    result = {
        "y_true": np.array(all_true),
        "y_score": np.array(all_score),
    }
    if has_logits and all_logits:
        result["logits"] = np.array(all_logits)

    return result


def probs_to_logits(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert probabilities to logits: log(p / (1 - p))."""
    probs = np.clip(probs, eps, 1 - eps)
    return np.log(probs / (1 - probs))


def nll_with_temperature(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood after temperature scaling."""
    scaled = logits / T
    # Numerically stable sigmoid + BCE
    log_probs = -np.logaddexp(0, -scaled)  # log(sigmoid(x))
    log_1m_probs = -np.logaddexp(0, scaled)  # log(1 - sigmoid(x))
    nll = -(labels * log_probs + (1 - labels) * log_1m_probs)
    return nll.mean()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find optimal temperature T that minimizes NLL on the data."""
    result = minimize_scalar(
        nll_with_temperature,
        bounds=(0.1, 20.0),
        args=(logits, labels),
        method="bounded",
    )
    return float(result.x)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)


def calibrate_model(
    model_name: str,
    data: dict,
    target_recall: float,
) -> dict:
    """Calibrate a single model: fit temperature, find threshold.

    Returns dict with temperature, threshold, and before/after metrics.
    """
    y_true = data["y_true"]
    y_score = data["y_score"]

    # Get logits: use saved logits if available, otherwise invert probs
    if "logits" in data:
        logits = data["logits"]
        logger.info("  Using saved logits")
    else:
        logits = probs_to_logits(y_score)
        logger.info("  Converting probs → logits (inverse sigmoid)")

    # Before calibration metrics
    ece_before = expected_calibration_error(y_true, y_score)
    metrics_before = compute_classification_metrics(y_true, y_score)
    thresh_before = find_threshold_at_recall(y_true, y_score, target_recall)
    flagged_before = float((y_score >= 0.5).mean())

    # Fit temperature
    T = fit_temperature(logits, y_true)
    logger.info("  Temperature T = %.4f", T)

    # Apply temperature scaling
    calibrated_probs = 1.0 / (1.0 + np.exp(-logits / T))

    # After calibration metrics
    ece_after = expected_calibration_error(y_true, calibrated_probs)
    metrics_after = compute_classification_metrics(y_true, calibrated_probs)
    thresh_after = find_threshold_at_recall(y_true, calibrated_probs, target_recall)
    flagged_after = float((calibrated_probs >= thresh_after).mean())

    # Print comparison
    logger.info("  ECE: %.4f → %.4f", ece_before, ece_after)
    logger.info("  AUPRC: %.4f → %.4f", metrics_before["auprc"], metrics_after["auprc"])
    logger.info(
        "  Threshold @%.0f%% recall: %.4f (before=%.4f)",
        target_recall * 100, thresh_after, thresh_before,
    )
    logger.info(
        "  Flagged %%: %.2f%% (at 0.5) → %.2f%% (at calibrated threshold)",
        flagged_before * 100, flagged_after * 100,
    )

    return {
        "model": model_name,
        "temperature": T,
        "threshold": thresh_after,
        "target_recall": target_recall,
        "before": {
            "ece": ece_before,
            "auprc": metrics_before["auprc"],
            "threshold_at_recall": thresh_before,
            "flagged_pct_at_0.5": flagged_before,
        },
        "after": {
            "ece": ece_after,
            "auprc": metrics_after["auprc"],
            "threshold_at_recall": thresh_after,
            "flagged_pct_at_threshold": flagged_after,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc calibration and threshold optimization"
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Root results directory containing model_a/, model_b/, model_c/ subdirs",
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.95,
        help="Target recall for threshold optimization (default: 0.95)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for calibration JSON (default: <results-dir>/calibration.json)",
    )
    args = parser.parse_args()

    output_path = args.output or args.results_dir / "calibration.json"

    model_names = ["model_a", "model_b", "model_c", "model_d"]
    calibration_results = {}

    for model_name in model_names:
        model_dir = args.results_dir / model_name
        if not model_dir.exists():
            logger.warning("Model dir %s not found, skipping", model_dir)
            continue

        logger.info("Calibrating %s...", model_name)
        data = load_fold_predictions(model_dir)
        if not data:
            logger.warning("No predictions found for %s, skipping", model_name)
            continue

        n_pos = int(data["y_true"].sum())
        n_neg = len(data["y_true"]) - n_pos
        logger.info("  Samples: %d (pos=%d, neg=%d)", len(data["y_true"]), n_pos, n_neg)

        result = calibrate_model(model_name, data, args.target_recall)
        calibration_results[model_name] = result

    if not calibration_results:
        logger.error("No models calibrated. Check that test_predictions.json exists in fold dirs.")
        return

    # Save calibration params
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration_results, f, indent=2)
    logger.info("Calibration saved to %s", output_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'T':>6} {'ECE Before':>12} {'ECE After':>12} {'Threshold':>10} {'Flagged %':>10}")
    print("-" * 70)
    for name, r in calibration_results.items():
        print(
            f"{name:<12} {r['temperature']:>6.3f} "
            f"{r['before']['ece']:>12.4f} {r['after']['ece']:>12.4f} "
            f"{r['threshold']:>10.4f} {r['after']['flagged_pct_at_threshold'] * 100:>9.2f}%"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
