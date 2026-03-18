"""CLI script to generate publication figures."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from blue_frogs.config import FIGURES_DIR, RESULTS_DIR
from blue_frogs.figures.paper_figures import (
    plot_precision_recall_comparison,
    plot_lab_distributions,
)
from blue_frogs.figures.supplementary_figures import (
    plot_reliability_diagram,
    plot_threshold_curves,
    plot_fold_boxplots,
    plot_confusion_matrix,
    plot_multi_model_comparison,
)
from blue_frogs.evaluation.metrics import find_threshold_at_recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_experiment_summary(summary_path: Path) -> dict | None:
    """Load experiment summary JSON with fold-level metrics."""
    if not summary_path.exists():
        logger.warning(f"Summary file not found: {summary_path}")
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_fold_predictions(model_dir: Path, n_folds: int = 5) -> dict | None:
    """Load predictions from all folds for a model."""
    all_scores = []
    y_true = None

    for fold in range(n_folds):
        pred_file = model_dir / f"fold_{fold}" / "test_predictions.json"
        if not pred_file.exists():
            continue
        with open(pred_file) as f:
            preds = json.load(f)
        all_scores.append(np.array(preds["y_score"]))
        if y_true is None:
            y_true = np.array(preds["y_true"])

    if not all_scores:
        return None

    return {
        "y_true": y_true,
        "y_score_ensemble": np.mean(all_scores, axis=0),
        "fold_scores": all_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--predictions-dir", type=Path, default=RESULTS_DIR,
                        help="Directory containing model prediction CSVs")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--summary-file", type=Path, default=None,
                        help="Path to experiment_summary.json for fold data")

    # Figure type flags
    parser.add_argument("--pr-curves", action="store_true",
                        help="Generate precision-recall curves")
    parser.add_argument("--reliability", action="store_true",
                        help="Generate reliability diagrams (calibration curves)")
    parser.add_argument("--threshold-curves", action="store_true",
                        help="Generate threshold curves (P/R/F1 vs threshold)")
    parser.add_argument("--fold-boxplots", action="store_true",
                        help="Generate per-fold variance box plots")
    parser.add_argument("--confusion", action="store_true",
                        help="Generate confusion matrix heatmaps")
    parser.add_argument("--comparison", action="store_true",
                        help="Generate model comparison bar charts")
    parser.add_argument("--all", action="store_true",
                        help="Generate all figure types")

    # Options
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for confusion matrix (default: optimal F1)")
    parser.add_argument("--recall-targets", type=float, nargs="+", default=[0.90, 0.95, 0.99],
                        help="Recall targets to mark on threshold curves")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # If no specific flags set, default to --all
    generate_all = args.all or not any([
        args.pr_curves, args.reliability, args.threshold_curves,
        args.fold_boxplots, args.confusion, args.comparison
    ])

    # Load experiment summary if available
    summary_path = args.summary_file or args.predictions_dir / "experiment_summary.json"
    summary = load_experiment_summary(summary_path)

    # Discover models and load predictions
    model_names = ["model_a", "model_b", "model_c"]
    model_data = {}
    model_scores = {}
    y_true = None

    for name in model_names:
        model_dir = args.predictions_dir / name
        if not model_dir.exists():
            continue
        data = load_fold_predictions(model_dir)
        if data:
            model_data[name] = data
            model_scores[name] = data["y_score_ensemble"]
            if y_true is None:
                y_true = data["y_true"]
            logger.info(f"Loaded {name}: {len(data['fold_scores'])} folds")

    # Also try loading from flat prediction CSVs (legacy format)
    for pred_file in sorted(args.predictions_dir.glob("predictions_*.csv")):
        model_name = pred_file.stem.replace("predictions_", "")
        if model_name in model_scores:
            continue
        df = pd.read_csv(pred_file)
        model_scores[model_name] = df["prediction_score"].values
        if y_true is None and "label" in df.columns:
            y_true = df["label"].values
        logger.info(f"Loaded {model_name} from CSV")

    if not model_scores:
        logger.error("No model predictions found")
        return

    # ===== Generate requested figures =====

    # 1. Precision-Recall curves
    if generate_all or args.pr_curves:
        if y_true is not None:
            pr_path = args.output_dir / "precision_recall_comparison.png"
            plot_precision_recall_comparison(y_true, model_scores, save_path=str(pr_path))
            logger.info(f"Saved PR curves to {pr_path}")

    # 2. Reliability diagrams (per model)
    if generate_all or args.reliability:
        for name, scores in model_scores.items():
            rel_path = args.output_dir / f"reliability_{name}.png"
            plot_reliability_diagram(y_true, scores, model_name=name, save_path=str(rel_path))
            logger.info(f"Saved reliability diagram to {rel_path}")

    # 3. Threshold curves (per model)
    if generate_all or args.threshold_curves:
        for name, scores in model_scores.items():
            thresh_path = args.output_dir / f"threshold_curves_{name}.png"
            plot_threshold_curves(
                y_true, scores,
                operating_points=args.recall_targets,
                model_name=name,
                save_path=str(thresh_path),
            )
            logger.info(f"Saved threshold curves to {thresh_path}")

    # 4. Per-fold box plots
    if generate_all or args.fold_boxplots:
        if summary and "models" in summary:
            for model_name, model_info in summary["models"].items():
                # Extract per-fold metrics from summary
                fold_metrics = {}
                for metric in ["auroc", "auprc", "f1_optimal"]:
                    if metric in model_info:
                        # We need actual fold values, not aggregated stats
                        # Load from fold predictions if available
                        pass

            # If we have fold data, create boxplots
            for name, data in model_data.items():
                if len(data["fold_scores"]) > 1:
                    from blue_frogs.evaluation.metrics import compute_classification_metrics
                    fold_metrics = {"auroc": [], "auprc": [], "f1_optimal": []}
                    for fold_scores in data["fold_scores"]:
                        m = compute_classification_metrics(data["y_true"], fold_scores)
                        fold_metrics["auroc"].append(m["auroc"])
                        fold_metrics["auprc"].append(m["auprc"])
                        fold_metrics["f1_optimal"].append(m["f1_optimal"])

                    box_path = args.output_dir / f"fold_boxplots_{name}.png"
                    plot_fold_boxplots(fold_metrics, save_path=str(box_path))
                    logger.info(f"Saved fold boxplots to {box_path}")

    # 5. Confusion matrices
    if generate_all or args.confusion:
        for name, scores in model_scores.items():
            # Determine threshold
            if args.threshold is not None:
                threshold = args.threshold
            else:
                # Use optimal F1 threshold
                from sklearn.metrics import precision_recall_curve
                precision, recall, thresholds = precision_recall_curve(y_true, scores)
                f1 = np.where(
                    (precision + recall) > 0,
                    2 * precision * recall / (precision + recall),
                    0,
                )
                threshold = thresholds[np.argmax(f1[:-1])]

            cm_path = args.output_dir / f"confusion_matrix_{name}.png"
            plot_confusion_matrix(
                y_true, scores,
                threshold=threshold,
                model_name=name,
                save_path=str(cm_path),
            )
            logger.info(f"Saved confusion matrix to {cm_path}")

    # 6. Model comparison
    if generate_all or args.comparison:
        if len(model_scores) > 1:
            for metric in ["auroc", "auprc"]:
                comp_path = args.output_dir / f"model_comparison_{metric}.png"
                plot_multi_model_comparison(
                    y_true, model_scores,
                    metric=metric,
                    save_path=str(comp_path),
                )
                logger.info(f"Saved model comparison to {comp_path}")

    logger.info("Figure generation complete")


if __name__ == "__main__":
    main()
