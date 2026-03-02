"""Aggregate per-fold results into ensemble predictions and comparison tables.

Run after all 5 folds complete for each model. Collects per-fold test predictions,
computes ensemble averages, and runs statistical comparison across models.

Usage:
    python scripts/aggregate_folds.py --results-dir results/ --output-dir results/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from blue_frogs.config import RESULTS_DIR
from blue_frogs.evaluation.comparison import compare_models, mcnemar_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_fold_predictions(
    model_dir: Path, n_folds: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Average test predictions across folds for ensemble evaluation.

    Each fold directory should contain test_predictions.json with:
        {"y_true": [...], "y_score": [...]}

    Returns (y_true, y_score_ensemble) where y_score_ensemble is the
    mean of per-fold predicted probabilities.
    """
    all_scores = []
    y_true = None

    for fold in range(n_folds):
        pred_file = model_dir / f"fold_{fold}" / "test_predictions.json"
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_file}")

        with open(pred_file) as f:
            preds = json.load(f)

        scores = np.array(preds["y_score"])
        all_scores.append(scores)

        if y_true is None:
            y_true = np.array(preds["y_true"])

    y_score_ensemble = np.mean(all_scores, axis=0)
    return y_true, y_score_ensemble


def generate_comparison_table(model_results: dict) -> str:
    """Generate a markdown comparison table from model results."""
    lines = [
        "| Model | AUPRC | AUPRC 95% CI | AUROC | AUROC 95% CI | F1 |",
        "|-------|-------|-------------|-------|-------------|-----|",
    ]
    for name, metrics in sorted(model_results.items()):
        auprc = metrics["auprc"]
        auprc_ci = f"[{metrics['auprc_ci_lower']:.4f}, {metrics['auprc_ci_upper']:.4f}]"
        auroc = metrics["auroc"]
        auroc_ci = f"[{metrics['auroc_ci_lower']:.4f}, {metrics['auroc_ci_upper']:.4f}]"
        f1 = metrics["f1_optimal"]
        lines.append(
            f"| {name} | {auprc:.4f} | {auprc_ci} | {auroc:.4f} | {auroc_ci} | {f1:.4f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate fold results and compare models")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing per-model result subdirectories")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = ["model_a", "model_b", "model_c"]

    # Aggregate per-model ensemble predictions
    model_scores = {}
    y_true = None
    for name in model_names:
        model_dir = args.results_dir / name
        if not model_dir.exists():
            logger.warning(f"Skipping {name} — no results directory")
            continue
        try:
            yt, ys = aggregate_fold_predictions(model_dir, args.n_folds)
            model_scores[name] = ys
            if y_true is None:
                y_true = yt
        except FileNotFoundError as e:
            logger.warning(f"Skipping {name}: {e}")

    if not model_scores:
        logger.error("No model results found")
        return

    # Compare all models
    logger.info(f"Comparing {len(model_scores)} models...")
    results = compare_models(y_true, model_scores)

    # Print and save comparison table
    table = generate_comparison_table(results)
    logger.info(f"\n{table}")

    table_path = args.output_dir / "comparison_summary.md"
    table_path.write_text(f"# Model Comparison\n\n{table}\n")
    logger.info(f"Table saved to {table_path}")

    # Pairwise McNemar's tests
    names = sorted(model_scores.keys())
    mcnemar_results = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            thresh_a = results[name_a]["optimal_threshold"]
            thresh_b = results[name_b]["optimal_threshold"]
            preds_a = (model_scores[name_a] >= thresh_a).astype(int)
            preds_b = (model_scores[name_b] >= thresh_b).astype(int)
            mc = mcnemar_test(y_true, preds_a, preds_b)
            pair = f"{name_a}_vs_{name_b}"
            mcnemar_results[pair] = mc
            logger.info(f"McNemar {pair}: p={mc['p_value']:.4f}")

    # Save full results
    output = {"model_metrics": results, "mcnemar_tests": mcnemar_results}
    json_path = args.output_dir / "comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
