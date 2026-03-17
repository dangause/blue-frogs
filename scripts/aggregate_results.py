"""Aggregate experiment results into a publication-ready summary.

Collects per-fold predictions, computes bootstrap confidence intervals,
and generates a unified experiment summary JSON for model cards and reporting.

Usage:
    python scripts/aggregate_results.py \
        --results-dir results/ \
        --calibration-file results/calibration.json \
        --output results/experiment_summary.json
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from blue_frogs.config import RESULTS_DIR
from blue_frogs.evaluation.metrics import (
    compute_bootstrap_ci,
    compute_classification_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_git_info() -> dict:
    """Get current git commit and branch info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def load_fold_predictions(model_dir: Path, n_folds: int = 5) -> dict:
    """Load test predictions from all completed folds.

    Returns dict with:
        - y_true: ground truth labels
        - y_score_ensemble: mean prediction across folds
        - fold_scores: list of per-fold predictions
        - n_folds_completed: number of folds found
    """
    all_scores = []
    y_true = None
    logits_available = True

    for fold in range(n_folds):
        pred_file = model_dir / f"fold_{fold}" / "test_predictions.json"
        if not pred_file.exists():
            logger.warning(f"Missing fold {fold} predictions: {pred_file}")
            continue

        with open(pred_file) as f:
            preds = json.load(f)

        scores = np.array(preds["y_score"])
        all_scores.append(scores)

        if y_true is None:
            y_true = np.array(preds["y_true"])

        if "logits" not in preds:
            logits_available = False

    if not all_scores:
        return None

    return {
        "y_true": y_true,
        "y_score_ensemble": np.mean(all_scores, axis=0),
        "fold_scores": all_scores,
        "n_folds_completed": len(all_scores),
        "logits_available": logits_available,
    }


def compute_fold_metrics(
    y_true: np.ndarray,
    fold_scores: list[np.ndarray],
    n_bootstrap: int = 1000,
) -> dict:
    """Compute metrics across folds with confidence intervals.

    Uses both:
    1. Per-fold metric variability (cross-validation CI)
    2. Bootstrap CIs on ensemble predictions
    """
    # Per-fold metrics
    fold_metrics = {"auroc": [], "auprc": [], "f1_optimal": []}
    for scores in fold_scores:
        m = compute_classification_metrics(y_true, scores)
        for k in fold_metrics:
            fold_metrics[k].append(m[k])

    # Compute mean and CI from fold variability
    result = {}
    for metric_name, values in fold_metrics.items():
        values = np.array(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        n = len(values)
        # t-based 95% CI (conservative for small n)
        from scipy.stats import t
        t_val = t.ppf(0.975, df=n - 1) if n > 1 else 2.0
        ci_half = t_val * std / np.sqrt(n) if n > 1 else 0
        result[metric_name] = {
            "mean": mean,
            "std": std,
            "ci_lower": mean - ci_half,
            "ci_upper": mean + ci_half,
            "n_folds": n,
        }

    # Also add bootstrap CIs on ensemble (for reference)
    ensemble_scores = np.mean(fold_scores, axis=0)
    for metric_name in ["auroc", "auprc"]:
        boot_ci = compute_bootstrap_ci(y_true, ensemble_scores, metric_name, n_bootstrap)
        result[metric_name]["bootstrap_ci_lower"] = boot_ci["lower"]
        result[metric_name]["bootstrap_ci_upper"] = boot_ci["upper"]

    return result


def aggregate_model_results(
    model_dir: Path,
    calibration: dict | None,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
) -> dict | None:
    """Aggregate results for a single model."""
    data = load_fold_predictions(model_dir, n_folds)
    if data is None:
        return None

    metrics = compute_fold_metrics(
        data["y_true"], data["fold_scores"], n_bootstrap
    )

    # Add calibration info if available
    model_name = model_dir.name
    if calibration and model_name in calibration:
        cal = calibration[model_name]
        metrics["calibration"] = {
            "temperature": cal.get("temperature"),
            "threshold": cal.get("threshold"),
            "ece_before": cal.get("ece_before"),
            "ece_after": cal.get("ece_after"),
        }

    # Add data stats
    y_true = data["y_true"]
    metrics["data_stats"] = {
        "n_test_samples": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(np.sum(1 - y_true)),
        "pos_ratio": float(np.mean(y_true)),
    }

    metrics["folds_completed"] = data["n_folds_completed"]
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into publication-ready summary"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory containing per-model result subdirectories",
    )
    parser.add_argument(
        "--calibration-file", type=Path, default=None,
        help="Path to calibration.json with per-model temperature and thresholds",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for experiment_summary.json",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    output_path = args.output or args.results_dir / "experiment_summary.json"

    # Load calibration if provided
    calibration = None
    if args.calibration_file and args.calibration_file.exists():
        with open(args.calibration_file) as f:
            calibration = json.load(f)
        logger.info(f"Loaded calibration from {args.calibration_file}")

    # Discover models
    model_names = ["model_a", "model_b", "model_c"]
    model_results = {}

    for name in model_names:
        model_dir = args.results_dir / name
        if not model_dir.exists():
            logger.warning(f"Skipping {name} - no results directory")
            continue

        result = aggregate_model_results(
            model_dir, calibration, args.n_folds, args.n_bootstrap
        )
        if result:
            model_results[name] = result
            logger.info(
                f"{name}: {result['folds_completed']}/{args.n_folds} folds, "
                f"AUPRC={result['auprc']['mean']:.4f} [{result['auprc']['ci_lower']:.4f}, {result['auprc']['ci_upper']:.4f}]"
            )

    if not model_results:
        logger.error("No model results found")
        return

    # Build summary
    git_info = get_git_info()
    summary = {
        "experiment_date": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "n_folds": args.n_folds,
        "n_bootstrap": args.n_bootstrap,
        "models": model_results,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Experiment summary saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<12} {'AUPRC':>12} {'95% CI':>24} {'Folds':>8}")
    print("-" * 60)
    for name, m in sorted(model_results.items()):
        auprc = m["auprc"]
        ci = f"[{auprc['ci_lower']:.4f}, {auprc['ci_upper']:.4f}]"
        print(f"{name:<12} {auprc['mean']:>12.4f} {ci:>24} {m['folds_completed']:>8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
