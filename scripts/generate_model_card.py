"""Generate model cards for trained axanthism classifiers.

Creates publication-ready model cards in markdown format with architecture details,
training configuration, performance metrics, and usage guidelines.

Usage:
    python scripts/generate_model_card.py \
        --model model_a \
        --config configs/model_a.yaml \
        --summary results/experiment_summary.json \
        --output results/model_a/MODEL_CARD.md

    # Generate for all models
    python scripts/generate_model_card.py --all --results-dir results/
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from blue_frogs.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = '''# Model Card: {model_name}

> Auto-generated on {generation_date}

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | {backbone} |
| **Training Strategy** | {training_strategy} |
| **Loss Function** | {loss_type} |
| **Optimizer** | {optimizer} (lr={learning_rate}) |
| **Image Size** | {image_size}x{image_size} |
| **Git Commit** | `{git_commit}` |

## Intended Use

Binary classification of **axanthism** (blue coloration) in Anura (frog/toad) images
from iNaturalist research-grade observations.

### Primary Use Case
- Automated screening of large photo databases to identify potential axanthic specimens
- Supporting biodiversity research on color mutation prevalence

### Out of Scope
- Medical diagnosis
- Species identification
- Images from non-iNaturalist sources (different lighting/quality)

## Training Data

| Split | Samples | Positive | Negative | Pos Ratio |
|-------|---------|----------|----------|-----------|
{data_table}

### Data Sources
- **Positives**: Confirmed axanthic observations from Womack et al. dataset
- **Negatives**: Stratified sample with hard/medium/easy tiers from iNaturalist

## Performance Metrics

### Cross-Validation Results ({n_folds}-fold CV)

| Metric | Mean | 95% CI | Std |
|--------|------|--------|-----|
{metrics_table}

### Calibration
{calibration_section}

## Limitations

1. **Domain Shift**: Trained on research-grade iNaturalist photos; performance may degrade on:
   - Low-resolution or blurry images
   - Unusual lighting conditions (artificial light, flash)
   - Non-standard camera angles

2. **Class Imbalance**: Axanthism is rare (~1-10% prevalence in training data). The model
   is optimized for high recall at the expense of precision.

3. **Taxonomic Scope**: Trained only on Anura (frogs and toads). Not validated for
   other amphibian orders or reptiles.

4. **Geographic Bias**: Training data may over-represent certain geographic regions
   based on iNaturalist observation density.

## Ethical Considerations

- Model predictions should be reviewed by domain experts before publication
- High-confidence predictions are candidates for verification, not confirmed diagnoses
- Consider collection pressure on wild populations when publicizing observation locations

## How to Use

```python
from blue_frogs.models.{model_module} import {model_class}
import torch

# Load model
model = {model_class}.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Predict
with torch.no_grad():
    logits = model(image_tensor)
    probability = torch.sigmoid(logits).item()

# Apply calibrated threshold
threshold = {threshold}  # calibrated for 95% recall
is_axanthic = probability >= threshold
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{blue_frogs_2026,
  title={{Blue Frogs: Deep Learning for Axanthism Detection}},
  author={{Author Name}},
  year={{2026}},
  howpublished={{\\url{{https://github.com/your-repo/blue-frogs}}}}
}}
```

## Model Card Contact

For questions or feedback, please open an issue on the project repository.
'''


def get_model_metadata(model_name: str) -> dict:
    """Return model-specific metadata."""
    metadata = {
        "model_a": {
            "backbone": "EfficientNetV2-S (ImageNet pretrained)",
            "model_module": "model_a",
            "model_class": "EfficientNetClassifier",
            "training_strategy": "2-stage: linear probe + full fine-tune",
        },
        "model_b": {
            "backbone": "EfficientNetV2-S + LAB color features (fusion)",
            "model_module": "model_b_classifier",
            "model_class": "FusionClassifier",
            "training_strategy": "Single-stage with weighted sampling",
        },
        "model_c": {
            "backbone": "DINOv2-Base (self-supervised pretrained)",
            "model_module": "model_c",
            "model_class": "FoundationModelClassifier",
            "training_strategy": "2-stage: linear probe + differential LR fine-tune",
        },
    }
    return metadata.get(model_name, {
        "backbone": "Unknown",
        "model_module": "unknown",
        "model_class": "Unknown",
        "training_strategy": "Unknown",
    })


def format_metrics_table(metrics: dict) -> str:
    """Format metrics dictionary as markdown table rows."""
    rows = []
    metric_order = ["auprc", "auroc", "f1_optimal"]
    metric_names = {"auprc": "AUPRC", "auroc": "AUROC", "f1_optimal": "F1 (optimal)"}

    for key in metric_order:
        if key not in metrics:
            continue
        m = metrics[key]
        name = metric_names.get(key, key)
        mean = m.get("mean", 0)
        ci_lo = m.get("ci_lower", mean)
        ci_hi = m.get("ci_upper", mean)
        std = m.get("std", 0)
        rows.append(f"| {name} | {mean:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | {std:.4f} |")

    return "\n".join(rows)


def format_calibration_section(calibration: dict | None) -> str:
    """Format calibration info as markdown."""
    if not calibration:
        return "Calibration data not available."

    lines = []
    if "temperature" in calibration and calibration["temperature"] is not None:
        lines.append(f"- **Temperature scaling**: T = {calibration['temperature']:.4f}")
    if "threshold" in calibration and calibration["threshold"] is not None:
        lines.append(f"- **Decision threshold**: {calibration['threshold']:.4f} (optimized for 95% recall)")
    if "ece_before" in calibration and calibration["ece_before"] is not None:
        lines.append(f"- **ECE before calibration**: {calibration['ece_before']:.4f}")
    if "ece_after" in calibration and calibration["ece_after"] is not None:
        lines.append(f"- **ECE after calibration**: {calibration['ece_after']:.4f}")

    return "\n".join(lines) if lines else "Calibration data not available."


def format_data_table(data_stats: dict | None) -> str:
    """Format data statistics as markdown table rows."""
    if not data_stats:
        return "| Test | - | - | - | - |"

    n_total = data_stats.get("n_test_samples", 0)
    n_pos = data_stats.get("n_positive", 0)
    n_neg = data_stats.get("n_negative", 0)
    pos_ratio = data_stats.get("pos_ratio", 0)

    return f"| Test | {n_total:,} | {n_pos:,} | {n_neg:,} | {pos_ratio:.2%} |"


def generate_model_card(
    model_name: str,
    config: dict,
    summary: dict | None,
    output_path: Path,
) -> None:
    """Generate model card markdown file."""
    meta = get_model_metadata(model_name)

    # Extract config values
    training = config.get("training", {})
    model_cfg = config.get("model", config.get("classifier", {}))
    data_cfg = config.get("data", {})

    # Get metrics from summary if available
    model_summary = (summary or {}).get("models", {}).get(model_name, {})
    calibration = model_summary.get("calibration", {})
    data_stats = model_summary.get("data_stats", {})

    # Fill template
    card_content = MODEL_CARD_TEMPLATE.format(
        model_name=model_name.replace("_", " ").title(),
        generation_date=datetime.utcnow().strftime("%Y-%m-%d"),
        backbone=meta["backbone"],
        training_strategy=meta["training_strategy"],
        loss_type=training.get("loss_type", "focal").replace("_", " ").title(),
        optimizer=training.get("optimizer", "AdamW").upper(),
        learning_rate=training.get("learning_rate", 0.001),
        image_size=data_cfg.get("image_size", 384),
        git_commit=(summary or {}).get("git_commit", "unknown")[:8],
        data_table=format_data_table(data_stats),
        n_folds=data_cfg.get("n_folds", 5),
        metrics_table=format_metrics_table(model_summary),
        calibration_section=format_calibration_section(calibration),
        model_module=meta["model_module"],
        model_class=meta["model_class"],
        threshold=calibration.get("threshold", 0.5) if calibration else 0.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card_content)
    logger.info(f"Model card saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model cards for trained classifiers"
    )
    parser.add_argument("--model", type=str, help="Model name (model_a, model_b, model_c)")
    parser.add_argument("--config", type=Path, help="Path to model config YAML")
    parser.add_argument("--summary", type=Path, help="Path to experiment_summary.json")
    parser.add_argument("--output", type=Path, help="Output path for MODEL_CARD.md")
    parser.add_argument(
        "--all", action="store_true",
        help="Generate model cards for all models",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Results directory (used with --all)",
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=Path("configs"),
        help="Configs directory (used with --all)",
    )
    args = parser.parse_args()

    # Load experiment summary if available
    summary = None
    summary_path = args.summary or args.results_dir / "experiment_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info(f"Loaded experiment summary from {summary_path}")

    if args.all:
        # Generate for all models
        model_names = ["model_a", "model_b", "model_c"]
        for model_name in model_names:
            config_path = args.configs_dir / f"{model_name}.yaml"
            if not config_path.exists():
                logger.warning(f"Skipping {model_name} - no config at {config_path}")
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f)

            output_path = args.results_dir / model_name / "MODEL_CARD.md"
            generate_model_card(model_name, config, summary, output_path)
    else:
        # Generate for single model
        if not args.model or not args.config:
            parser.error("--model and --config are required (or use --all)")

        with open(args.config) as f:
            config = yaml.safe_load(f)

        output_path = args.output or args.results_dir / args.model / "MODEL_CARD.md"
        generate_model_card(args.model, config, summary, output_path)


if __name__ == "__main__":
    main()
