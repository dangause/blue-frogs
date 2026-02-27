"""Publication-quality figures for the axanthism classifier paper."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
})


def plot_precision_recall_comparison(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    save_path: str | None = None,
):
    """Plot precision-recall curves for all models on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"model_a": "#1f77b4", "model_b": "#ff7f0e", "model_c": "#2ca02c"}
    labels = {"model_a": "EfficientNetV2-S", "model_b": "Two-Stage + LAB", "model_c": "DINOv2"}

    for name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        auprc = np.trapz(precision, recall)
        ax.plot(
            recall, precision,
            color=colors.get(name, "gray"),
            label=f"{labels.get(name, name)} (AUPRC={auprc:.3f})",
            linewidth=2,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_lab_distributions(
    positive_features: np.ndarray,
    negative_features: np.ndarray,
    save_path: str | None = None,
):
    """Plot LAB b* channel distributions for axanthic vs. normal frogs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channel_names = ["L* (Luminance)", "a* (Green-Red)", "b* (Blue-Yellow)"]
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        offset = i * 10  # 10 features per channel, mean is index 0
        pos_mean = positive_features[:, offset]
        neg_mean = negative_features[:, offset]

        ax.hist(neg_mean, bins=30, alpha=0.5, label="Normal", color="#2ca02c", density=True)
        ax.hist(pos_mean, bins=30, alpha=0.5, label="Axanthic", color="#1f77b4", density=True)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("LAB Color Channel Distributions")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
