"""CLI script to compare trained models on the held-out test set."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from blue_frogs.config import RESULTS_DIR
from blue_frogs.evaluation.comparison import compare_models, mcnemar_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compare model predictions on test set")
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="Directory containing per-model prediction CSVs")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions from each model
    model_scores = {}
    y_true = None
    for pred_file in sorted(args.predictions_dir.glob("*.csv")):
        model_name = pred_file.stem
        df = pd.read_csv(pred_file)
        model_scores[model_name] = df["prediction_score"].values
        if y_true is None:
            y_true = df["label"].values

    if not model_scores:
        logger.error("No prediction files found")
        return

    # Compare all models
    results = compare_models(y_true, model_scores)
    logger.info("Model comparison results:")
    for name, metrics in results.items():
        logger.info(
            f"  {name}: AUPRC={metrics['auprc']:.4f} "
            f"[{metrics['auprc_ci_lower']:.4f}, {metrics['auprc_ci_upper']:.4f}], "
            f"AUROC={metrics['auroc']:.4f}"
        )

    # Pairwise McNemar's tests
    model_names = sorted(model_scores.keys())
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            threshold_a = results[name_a]["optimal_threshold"]
            threshold_b = results[name_b]["optimal_threshold"]
            preds_a = (model_scores[name_a] >= threshold_a).astype(int)
            preds_b = (model_scores[name_b] >= threshold_b).astype(int)
            mc = mcnemar_test(y_true, preds_a, preds_b)
            logger.info(f"  McNemar {name_a} vs {name_b}: p={mc['p_value']:.4f}")

    # Save results
    output_path = args.output_dir / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
