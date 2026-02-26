"""CLI script to generate publication figures."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from blue_frogs.config import FIGURES_DIR, RESULTS_DIR
from blue_frogs.figures.paper_figures import (
    plot_precision_recall_comparison,
    plot_lab_distributions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--predictions-dir", type=Path, default=RESULTS_DIR,
                        help="Directory containing model prediction CSVs")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    model_scores = {}
    y_true = None
    for pred_file in sorted(args.predictions_dir.glob("predictions_*.csv")):
        model_name = pred_file.stem.replace("predictions_", "")
        df = pd.read_csv(pred_file)
        model_scores[model_name] = df["prediction_score"].values
        if y_true is None and "label" in df.columns:
            y_true = df["label"].values

    if model_scores and y_true is not None:
        # Precision-Recall curves
        pr_path = args.output_dir / "precision_recall_comparison.png"
        plot_precision_recall_comparison(y_true, model_scores, save_path=str(pr_path))
        logger.info(f"Saved PR curves to {pr_path}")

    logger.info("Figure generation complete")


if __name__ == "__main__":
    main()
