"""Generate supplementary materials tables for publication appendix.

Outputs:
1. Per-fold metrics table (AUROC, AUPRC, F1, threshold per fold)
2. Hyperparameter table (consolidated from configs)
3. McNemar's test matrix (pairwise p-values)
4. Threshold operating points (precision/flagged% at various recall levels)

Usage:
    python scripts/generate_supplementary.py \
        --results-dir results/ \
        --configs-dir configs/ \
        --output-dir docs/supplementary/
"""

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from blue_frogs.config import RESULTS_DIR
from blue_frogs.evaluation.comparison import mcnemar_test
from blue_frogs.evaluation.metrics import (
    compute_classification_metrics,
    find_threshold_at_recall,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def generate_fold_metrics_table(model_data: dict, output_dir: Path):
    """Generate per-fold metrics table in Markdown and LaTeX."""
    rows = []

    for model_name, data in sorted(model_data.items()):
        y_true = data["y_true"]
        for fold_idx, fold_scores in enumerate(data["fold_scores"]):
            metrics = compute_classification_metrics(y_true, fold_scores)
            rows.append({
                "Model": model_name,
                "Fold": fold_idx,
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "F1": metrics["f1_optimal"],
                "Threshold": metrics["optimal_threshold"],
            })

    # Markdown table
    md_lines = [
        "# Per-Fold Metrics",
        "",
        "| Model | Fold | AUROC | AUPRC | F1 | Threshold |",
        "|-------|------|-------|-------|----|-----------| ",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['Model']} | {r['Fold']} | {r['AUROC']:.4f} | {r['AUPRC']:.4f} | "
            f"{r['F1']:.4f} | {r['Threshold']:.4f} |"
        )

    # Add summary statistics
    md_lines.extend(["", "## Summary (Mean ± Std)", ""])
    for model_name in sorted(model_data.keys()):
        model_rows = [r for r in rows if r["Model"] == model_name]
        for metric in ["AUROC", "AUPRC", "F1"]:
            values = [r[metric] for r in model_rows]
            mean, std = np.mean(values), np.std(values, ddof=1)
            md_lines.append(f"- **{model_name}** {metric}: {mean:.4f} ± {std:.4f}")

    md_path = output_dir / "fold_metrics.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved fold metrics Markdown to {md_path}")

    # LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-Fold Classification Metrics}",
        r"\label{tab:fold-metrics}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Fold & AUROC & AUPRC & F1 & Threshold \\",
        r"\midrule",
    ]
    for r in rows:
        latex_lines.append(
            f"{r['Model']} & {r['Fold']} & {r['AUROC']:.4f} & {r['AUPRC']:.4f} & "
            f"{r['F1']:.4f} & {r['Threshold']:.4f} \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = output_dir / "fold_metrics.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"Saved fold metrics LaTeX to {latex_path}")


def generate_hyperparameter_table(configs_dir: Path, output_dir: Path):
    """Generate consolidated hyperparameter table from config files."""
    config_files = list(configs_dir.glob("model_*.yaml"))
    if not config_files:
        logger.warning("No config files found")
        return

    # Collect hyperparameters
    all_params = {}
    param_keys = set()

    for config_file in sorted(config_files):
        model_name = config_file.stem
        with open(config_file) as f:
            config = yaml.safe_load(f)

        params = {}

        # Extract common training params
        if "training" in config:
            t = config["training"]
            params["Batch Size"] = t.get("batch_size")
            params["Learning Rate"] = t.get("learning_rate")
            params["Weight Decay"] = t.get("weight_decay")
            params["Loss Type"] = t.get("loss_type")
            params["Max Epochs"] = t.get("max_epochs")
            params["Early Stopping"] = t.get("early_stopping_patience")

        # Model-specific params
        if "model" in config:
            m = config["model"]
            params["Backbone"] = m.get("backbone")
            params["Dropout"] = m.get("dropout")
            params["Hidden Dim"] = m.get("hidden_dim")

        if "classifier" in config:
            c = config["classifier"]
            params["Backbone"] = c.get("backbone")
            params["Dropout"] = c.get("dropout")
            params["Fusion Hidden"] = c.get("fusion_hidden")

        all_params[model_name] = params
        param_keys.update(params.keys())

    # Sort parameter keys for consistent ordering
    param_keys = sorted(param_keys)

    # Markdown table
    md_lines = [
        "# Hyperparameters",
        "",
        "| Parameter | " + " | ".join(sorted(all_params.keys())) + " |",
        "|-----------|" + "|".join(["---"] * len(all_params)) + "|",
    ]

    for key in param_keys:
        values = [str(all_params[m].get(key, "-")) for m in sorted(all_params.keys())]
        md_lines.append(f"| {key} | " + " | ".join(values) + " |")

    md_path = output_dir / "hyperparameters.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved hyperparameters Markdown to {md_path}")

    # LaTeX table
    model_names = sorted(all_params.keys())
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Model Hyperparameters}",
        r"\label{tab:hyperparameters}",
        r"\begin{tabular}{l" + "c" * len(model_names) + "}",
        r"\toprule",
        r"Parameter & " + " & ".join(model_names) + r" \\",
        r"\midrule",
    ]

    for key in param_keys:
        values = [str(all_params[m].get(key, "-")).replace("_", r"\_") for m in model_names]
        latex_lines.append(f"{key} & " + " & ".join(values) + r" \\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = output_dir / "hyperparameters.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"Saved hyperparameters LaTeX to {latex_path}")


def generate_mcnemar_matrix(model_data: dict, output_dir: Path):
    """Generate pairwise McNemar's test matrix."""
    if len(model_data) < 2:
        logger.warning("Need at least 2 models for McNemar's test")
        return

    model_names = sorted(model_data.keys())
    y_true = next(iter(model_data.values()))["y_true"]

    # Compute pairwise tests
    results = {}
    for m1, m2 in combinations(model_names, 2):
        scores1 = model_data[m1]["y_score_ensemble"]
        scores2 = model_data[m2]["y_score_ensemble"]

        # Binarize at optimal threshold for each model
        metrics1 = compute_classification_metrics(y_true, scores1)
        metrics2 = compute_classification_metrics(y_true, scores2)

        preds1 = (scores1 >= metrics1["optimal_threshold"]).astype(int)
        preds2 = (scores2 >= metrics2["optimal_threshold"]).astype(int)

        test_result = mcnemar_test(y_true, preds1, preds2)
        results[(m1, m2)] = test_result

    # Markdown table
    md_lines = [
        "# McNemar's Test (Pairwise)",
        "",
        "P-values for pairwise comparison of model predictions (at optimal thresholds).",
        "",
        "| Model A | Model B | Statistic | p-value | Discordant |",
        "|---------|---------|-----------|---------|------------|",
    ]

    for (m1, m2), r in results.items():
        sig = "**" if r["p_value"] < 0.05 else ""
        md_lines.append(
            f"| {m1} | {m2} | {r['statistic']:.2f} | {sig}{r['p_value']:.4f}{sig} | {r['n_discordant']} |"
        )

    md_path = output_dir / "mcnemar_test.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved McNemar's test Markdown to {md_path}")

    # LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{McNemar's Test for Pairwise Model Comparison}",
        r"\label{tab:mcnemar}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model A & Model B & Statistic & p-value & Discordant \\",
        r"\midrule",
    ]

    for (m1, m2), r in results.items():
        pval = f"{r['p_value']:.4f}"
        if r["p_value"] < 0.05:
            pval = r"\textbf{" + pval + "}"
        latex_lines.append(
            f"{m1} & {m2} & {r['statistic']:.2f} & {pval} & {r['n_discordant']} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = output_dir / "mcnemar_test.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"Saved McNemar's test LaTeX to {latex_path}")


def generate_operating_points_table(model_data: dict, output_dir: Path):
    """Generate threshold operating points at various recall levels."""
    recall_targets = [0.90, 0.95, 0.99]

    rows = []
    for model_name, data in sorted(model_data.items()):
        y_true = data["y_true"]
        y_score = data["y_score_ensemble"]

        for target in recall_targets:
            threshold = find_threshold_at_recall(y_true, y_score, target)
            preds = (y_score >= threshold).astype(int)

            # Compute actual metrics at this threshold
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            actual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            flagged_pct = np.mean(preds)

            rows.append({
                "Model": model_name,
                "Target Recall": f"{target:.0%}",
                "Actual Recall": actual_recall,
                "Precision": precision,
                "Threshold": threshold,
                "Flagged %": flagged_pct,
            })

    # Markdown table
    md_lines = [
        "# Operating Points",
        "",
        "Precision and flagged percentage at various recall targets.",
        "",
        "| Model | Target | Actual Recall | Precision | Threshold | Flagged % |",
        "|-------|--------|---------------|-----------|-----------|-----------|",
    ]

    for r in rows:
        md_lines.append(
            f"| {r['Model']} | {r['Target Recall']} | {r['Actual Recall']:.1%} | "
            f"{r['Precision']:.1%} | {r['Threshold']:.4f} | {r['Flagged %']:.1%} |"
        )

    md_path = output_dir / "operating_points.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved operating points Markdown to {md_path}")

    # LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Operating Points at Various Recall Targets}",
        r"\label{tab:operating-points}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Target & Recall & Precision & Threshold & Flagged \% \\",
        r"\midrule",
    ]

    for r in rows:
        latex_lines.append(
            f"{r['Model']} & {r['Target Recall']} & {r['Actual Recall']:.1%} & "
            f"{r['Precision']:.1%} & {r['Threshold']:.4f} & {r['Flagged %']:.1%} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = output_dir / "operating_points.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"Saved operating points LaTeX to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate supplementary materials")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Directory containing model results")
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"),
                        help="Directory containing model config YAML files")
    parser.add_argument("--output-dir", type=Path, default=Path("docs") / "supplementary",
                        help="Output directory for supplementary materials")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model predictions
    model_names = ["model_a", "model_b", "model_c"]
    model_data = {}

    for name in model_names:
        model_dir = args.results_dir / name
        if not model_dir.exists():
            continue
        data = load_fold_predictions(model_dir, args.n_folds)
        if data:
            model_data[name] = data
            logger.info(f"Loaded {name}: {len(data['fold_scores'])} folds")

    if not model_data:
        logger.error("No model predictions found")
        return

    # Generate all tables
    generate_fold_metrics_table(model_data, args.output_dir)
    generate_hyperparameter_table(args.configs_dir, args.output_dir)
    generate_mcnemar_matrix(model_data, args.output_dir)
    generate_operating_points_table(model_data, args.output_dir)

    logger.info(f"All supplementary materials saved to {args.output_dir}")


if __name__ == "__main__":
    main()
