"""Supplementary figures for the axanthism classifier paper.

These figures complement the main PR curves and LAB distributions with:
- Reliability diagrams (calibration curves)
- Threshold curves (precision/recall/F1 vs threshold)
- Per-fold box plots (cross-validation variance)
- Confusion matrix heatmaps
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix

matplotlib.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
})


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = "Model",
    save_path: str | None = None,
):
    """Plot reliability diagram (calibration curve).

    Shows predicted probability vs actual frequency of positives.
    Perfect calibration lies on the diagonal.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        model_name: Name for plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1.5)

    # Plot model calibration
    ax.plot(prob_pred, prob_true, "s-", color="#1f77b4", label=model_name, linewidth=2, markersize=8)

    # Add histogram of predictions at bottom
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), alpha=0.3, color="#1f77b4", density=True)
    ax2.set_ylabel("Density", color="#1f77b4", alpha=0.7)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 3)  # Scale down histogram

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Diagram - {model_name}")
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_threshold_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    operating_points: list[float] | None = None,
    model_name: str = "Model",
    save_path: str | None = None,
):
    """Plot precision, recall, and F1 as functions of decision threshold.

    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores/probabilities
        operating_points: Optional list of recall targets to mark (e.g., [0.90, 0.95, 0.99])
        model_name: Name for plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Trim last element (precision/recall have len = thresholds + 1)
    precision = precision[:-1]
    recall = recall[:-1]

    # Compute F1 at each threshold
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )

    # Plot curves
    ax.plot(thresholds, precision, label="Precision", color="#2ca02c", linewidth=2)
    ax.plot(thresholds, recall, label="Recall", color="#1f77b4", linewidth=2)
    ax.plot(thresholds, f1_scores, label="F1 Score", color="#ff7f0e", linewidth=2)

    # Mark operating points if provided
    if operating_points:
        colors = ["#d62728", "#9467bd", "#8c564b"]
        for i, target_recall in enumerate(operating_points):
            valid_idx = np.where(recall >= target_recall)[0]
            if len(valid_idx) > 0:
                # Pick threshold with highest precision at this recall
                best_idx = valid_idx[np.argmax(precision[valid_idx])]
                thresh = thresholds[best_idx]
                prec = precision[best_idx]
                ax.axvline(thresh, color=colors[i % len(colors)], linestyle=":", alpha=0.7)
                ax.annotate(
                    f"R={target_recall:.0%}\nT={thresh:.3f}\nP={prec:.1%}",
                    xy=(thresh, 0.5),
                    xytext=(thresh + 0.05, 0.5 - i * 0.15),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=colors[i % len(colors)]),
                )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision/Recall/F1 vs Threshold - {model_name}")
    ax.legend(loc="center left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_fold_boxplots(
    fold_metrics: dict[str, list[float]],
    metric_labels: dict[str, str] | None = None,
    save_path: str | None = None,
):
    """Plot box plots showing cross-validation variance for each metric.

    Args:
        fold_metrics: Dict mapping metric name -> list of per-fold values
            e.g., {"auroc": [0.95, 0.94, 0.96, 0.95, 0.93], "auprc": [...]}
        metric_labels: Optional dict mapping metric name -> display label
        save_path: Optional path to save figure
    """
    if metric_labels is None:
        metric_labels = {
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "f1_optimal": "F1 (optimal)",
        }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data for boxplot
    data = []
    labels = []
    for metric_name in fold_metrics:
        values = fold_metrics[metric_name]
        data.append(values)
        label = metric_labels.get(metric_name, metric_name)
        labels.append(label)

    # Create boxplot
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#1f77b4", alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="#d62728", markersize=8),
    )

    # Add individual points
    for i, values in enumerate(data):
        x = np.random.normal(i + 1, 0.04, len(values))
        ax.scatter(x, values, alpha=0.5, color="#ff7f0e", s=50, zorder=3)

    ax.set_ylabel("Score")
    ax.set_title("Cross-Validation Variance by Metric")
    ax.grid(True, axis="y", alpha=0.3)

    # Add mean and std annotations
    for i, values in enumerate(data):
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ax.annotate(
            f"{mean:.3f} ± {std:.3f}",
            xy=(i + 1, mean),
            xytext=(i + 1.3, mean),
            fontsize=9,
            va="center",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None = None,
    model_name: str = "Model",
    class_names: list[str] | None = None,
    save_path: str | None = None,
):
    """Plot confusion matrix as a heatmap.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted labels (binary) or probabilities
        threshold: If y_pred are probabilities, threshold to binarize
        model_name: Name for plot title
        class_names: Names for classes (default: ["Normal", "Axanthic"])
        save_path: Optional path to save figure
    """
    if class_names is None:
        class_names = ["Normal", "Axanthic"]

    # Binarize if threshold provided
    if threshold is not None:
        y_pred = (np.array(y_pred) >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))

    # Create heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    # Add labels
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
    )

    # Add text annotations
    thresh_color = cm.max() / 2
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            total = cm[i].sum()
            pct = count / total * 100 if total > 0 else 0
            color = "white" if cm[i, j] > thresh_color else "black"
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=color, fontsize=14, fontweight="bold",
            )

    # Compute metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    title = f"Confusion Matrix - {model_name}"
    if threshold is not None:
        title += f" (threshold={threshold:.3f})"
    ax.set_title(title)

    # Add metrics text below
    metrics_text = f"Precision: {precision:.1%}  |  Recall: {recall:.1%}  |  Specificity: {specificity:.1%}"
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_multi_model_comparison(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    metric: str = "auprc",
    save_path: str | None = None,
):
    """Plot bar chart comparing models with error bars from bootstrap CI.

    Args:
        y_true: Ground truth binary labels
        model_scores: Dict mapping model name -> predicted scores
        metric: Metric to compare ("auroc" or "auprc")
        save_path: Optional path to save figure
    """
    from blue_frogs.evaluation.metrics import compute_bootstrap_ci

    # Friendly model names
    model_labels = {
        "model_a": "EfficientNetV2-S",
        "model_b": "Two-Stage + LAB",
        "model_c": "DINOv2",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    model_names = list(model_scores.keys())
    means = []
    ci_lower = []
    ci_upper = []

    for name in model_names:
        scores = model_scores[name]
        ci = compute_bootstrap_ci(y_true, scores, metric=metric, n_bootstrap=1000)
        means.append(ci["mean"])
        ci_lower.append(ci["mean"] - ci["lower"])
        ci_upper.append(ci["upper"] - ci["mean"])

    x = np.arange(len(model_names))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(
        x, means,
        yerr=[ci_lower, ci_upper],
        capsize=5,
        color=colors[:len(model_names)],
        alpha=0.8,
        edgecolor="black",
    )

    # Use friendly labels
    display_names = [model_labels.get(n, n) for n in model_names]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison - {metric.upper()}")

    # Dynamic y-axis: start from min - margin, end at 1.0 + margin for labels
    min_val = min(m - lo for m, lo in zip(means, ci_lower))
    ax.set_ylim([max(0, min_val - 0.05), 1.05])
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels inside bars (near top)
    for i, (m, lo, hi) in enumerate(zip(means, ci_lower, ci_upper)):
        # Place text inside the bar
        ax.text(
            i, m - 0.02,
            f"{m:.3f}",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
        )
        # Place CI below the bar
        ax.text(
            i, min_val - 0.02,
            f"95% CI: [{m-lo:.3f}, {m+hi:.3f}]",
            ha="center", va="top",
            fontsize=8, color="gray",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
