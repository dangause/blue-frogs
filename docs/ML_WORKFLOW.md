# Machine Learning Workflow

Comprehensive guide to the axanthism detection ML pipeline, from raw data to publication-ready results.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Architectures](#model-architectures)
4. [Training Pipeline](#training-pipeline)
5. [Calibration](#calibration)
6. [Inference](#inference)
7. [Evaluation](#evaluation)
8. [Publication Materials](#publication-materials)
9. [Reproducibility](#reproducibility)

---

## Overview

This pipeline detects **axanthism** (blue coloration mutation) in frog images from iNaturalist. The workflow consists of:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Data   │───▶│  Training   │───▶│ Calibration │───▶│  Inference  │
│  Pipeline   │    │  Pipeline   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  labels.json        checkpoints      calibration.json    predictions
  splits.json        fold metrics     thresholds          flagged images
```

### Key Design Decisions

- **Extreme class imbalance**: ~0.09% prevalence in wild iNaturalist corpus
- **High-recall priority**: False negatives (missed axanthic frogs) are worse than false positives
- **Ensemble approach**: Multiple architectures for robustness
- **Post-hoc calibration**: Temperature scaling to improve probability estimates

---

## Data Pipeline

### 1. Data Sources

| Source | Description | Count |
|--------|-------------|-------|
| Womack et al. CSV | Verified axanthic records | ~372 observations |
| iNaturalist API | Research-grade frog photos | 10M+ available |

### 2. Label Generation

```bash
python scripts/prepare_training_data.py \
    --data-dir data/ \
    --labels-csv data/womack_labels.csv
```

**Outputs:**
- `data/labels.json` - Photo metadata with labels
- `data/splits.json` - Train/test indices

### 3. Negative Sampling Strategy

Negatives are stratified into three difficulty tiers:

| Tier | Description | Default Count |
|------|-------------|---------------|
| **Hard** | Naturally blue species (*Dendrobates azureus*, etc.) | 1,000 |
| **Medium** | Same species as positives, normal coloration | 2,000 |
| **Easy** | Random frog species | 7,000 |

This tiered approach ensures the model learns to distinguish true axanthism from:
- Natural blue coloration (hard negatives)
- Intra-species color variation (medium negatives)
- General frog appearance (easy negatives)

### 4. Train/Test Split

- **Test set**: 20% held out, stratified by label
- **Cross-validation**: 5-fold stratified CV on training set
- **Stratification unit**: Observation ID (not photo ID) to prevent data leakage

### 5. Data Augmentation

```python
transforms = A.Compose([
    A.RandomResizedCrop(384, 384, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.014,  # ±5 degrees - constrained to preserve color signal
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

**Critical**: Hue jitter is constrained to ±5 degrees because axanthism is fundamentally a color-based phenotype.

---

## Model Architectures

### Model A: EfficientNetV2-S

**Architecture**: Standard CNN with ImageNet pretrained backbone

```
Input (384x384x3)
    │
    ▼
EfficientNetV2-S (pretrained)
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout (0.3)
    │
    ▼
Linear (1280 → 1)
    │
    ▼
Sigmoid → P(axanthic)
```

**Config**: `configs/model_a.yaml`

**Strengths**:
- Fast training and inference
- Strong baseline performance
- Well-understood architecture

### Model B: Two-Stage + LAB Fusion

**Architecture**: YOLO detection → CNN classification with explicit color features

```
Stage 1: Detection
┌─────────────────────────┐
│  YOLOv8n Frog Detector  │
│  (pretrained + fine-tuned)│
└───────────┬─────────────┘
            │ crop
            ▼
Stage 2: Classification
┌─────────────────────────┐
│   Input: Cropped Frog   │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌────────┐    ┌──────────┐
│  CNN   │    │   LAB    │
│backbone│    │ features │
│(ResNet)│    │ (30-dim) │
└───┬────┘    └────┬─────┘
    │              │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Fusion MLP   │
    │ (512 → 256   │
    │  → 128 → 1)  │
    └──────────────┘
           │
           ▼
      P(axanthic)
```

**LAB Features (30-dimensional)**:
- Per-channel (L*, a*, b*): mean, std, skew, kurtosis, min, max, percentiles (10, 25, 50, 75, 90)
- Pre-computed for efficiency: `data/crop_lab_features.npy`

**Config**: `configs/model_b.yaml`

**Strengths**:
- Explicit color reasoning via LAB features
- Focus on frog region (ignores background)
- Interpretable color statistics

### Model C: DINOv2 Foundation Model

**Architecture**: Self-supervised vision transformer with two-stage training

```
Stage 1: Linear Probe (backbone frozen)
┌─────────────────────────┐
│  DINOv2-Base (frozen)   │
│  (86M params)           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Linear Head (trainable)│
│  768 → 256 → 1          │
└─────────────────────────┘

Stage 2: Fine-tune (backbone unfrozen)
┌─────────────────────────┐
│  DINOv2-Base (unfrozen) │
│  LR: 1e-5 (backbone)    │
│  LR: 1e-4 (head)        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Linear Head            │
│  (initialized from      │
│   Stage 1 weights)      │
└─────────────────────────┘
```

**Config**: `configs/model_c.yaml`

**Strengths**:
- Best overall performance (AUPRC ~0.994)
- Rich visual representations from self-supervised pretraining
- Handles diverse image conditions well

---

## Training Pipeline

### Single Model Training

```bash
python scripts/train.py \
    --config configs/model_a.yaml \
    --model model_a \
    --data-dir data/ \
    --labels-file data/labels.json \
    --splits-file data/splits.json \
    --output-dir results/
```

### Training All Models (5-fold CV)

```bash
# On Spark/HPC node
bash scripts/spark/train_all.sh
```

This trains all three models across 5 folds with:
- Weighted BCE loss (pos_weight based on class ratio)
- AdamW optimizer with cosine annealing
- Early stopping on validation AUPRC
- WandB logging (offline mode for HPC)

### Output Structure

```
results/
├── model_a/
│   ├── fold_0/
│   │   ├── best-val/auprc=0.9664.ckpt
│   │   ├── test_predictions.json
│   │   └── training_metadata.json
│   ├── fold_1/
│   │   └── ...
│   └── training_summary.json
├── model_b/
│   └── ...
├── model_c/
│   └── ...
└── experiment_summary.json
```

### Training Metadata

Each fold saves comprehensive metadata for reproducibility:

```json
{
  "model_name": "model_a",
  "fold": 0,
  "timestamp": "2026-03-15T10:30:00Z",
  "git": {"commit": "abc123", "branch": "main", "dirty": false},
  "random_seeds": {
    "base_seed": 42,
    "fold_seed": 42,
    "numpy_seed": 42,
    "torch_seed": 42
  },
  "config": {...},
  "data_stats": {
    "n_train": 2112,
    "n_val": 529,
    "pos_ratio_train": 0.193
  },
  "environment": {
    "python_version": "3.11.0",
    "pytorch_version": "2.1.0",
    "cuda_version": "12.1"
  }
}
```

---

## Calibration

Raw neural network outputs are often overconfident. Post-hoc calibration improves probability estimates.

### Temperature Scaling

```bash
python scripts/calibrate.py \
    --results-dir results/ \
    --output results/calibration.json \
    --target-recall 0.95
```

**Process**:
1. Load validation predictions from all folds
2. Optimize temperature T to minimize ECE (Expected Calibration Error)
3. Compute calibrated probabilities: `p_calibrated = sigmoid(logit / T)`
4. Find threshold achieving target recall (e.g., 95%)

### Output

```json
{
  "model_a": {
    "temperature": 1.234,
    "threshold": 0.250,
    "ece_before": 0.089,
    "ece_after": 0.032,
    "precision_at_threshold": 0.868,
    "recall_at_threshold": 0.950
  }
}
```

---

## Inference

### Batch Inference (Known Dataset)

```bash
python scripts/run_inference.py \
    --model model_a \
    --checkpoint results/model_a/fold_0/best-val/auprc=0.9664.ckpt \
    --data-dir data/ \
    --output predictions.csv
```

### Streaming Inference (iNaturalist Corpus)

For scanning the full iNaturalist frog corpus without downloading all images:

```bash
python scripts/stream_inference.py \
    --models model_a model_c \
    --checkpoints results/model_a/fold_0/best.ckpt results/model_c/fold_0/best.ckpt \
    --calibration results/calibration.json \
    --taxon-id 20979 \
    --obs-per-batch 500 \
    --max-batches 100 \
    --output-dir results/streaming/
```

**Features**:
- Downloads images on-the-fly (cached)
- Multi-model ensemble scoring
- Resumes from last processed batch
- Saves flagged observations (above threshold)

### Streaming Output

```
results/streaming/
├── state.json              # Checkpoint for resumption
├── flagged_model_a.csv     # Observations above threshold
├── flagged_model_c.csv
└── batch_logs/
    ├── batch_001.json
    └── ...
```

---

## Evaluation

### Metrics

| Metric | Description | Why Used |
|--------|-------------|----------|
| **AUPRC** | Area Under Precision-Recall Curve | Primary metric for imbalanced data |
| **AUROC** | Area Under ROC Curve | Secondary, for comparison |
| **F1** | Harmonic mean of precision/recall | At optimal threshold |
| **Precision @ Recall** | Precision at fixed recall levels | Operational planning |

### Aggregate Results

```bash
python scripts/aggregate_results.py \
    --results-dir results/ \
    --calibration-file results/calibration.json \
    --output results/experiment_summary.json
```

### Model Comparison

```bash
python scripts/compare_models.py \
    --results-dir results/ \
    --output results/comparison.md
```

Outputs McNemar's test for pairwise statistical comparison.

### Expected Results

| Model | AUPRC | AUROC | F1 (optimal) |
|-------|-------|-------|--------------|
| EfficientNetV2-S | 0.975 ± 0.010 | 0.991 ± 0.004 | 0.909 ± 0.013 |
| Two-Stage + LAB | 0.974 ± 0.012 | 0.992 ± 0.003 | 0.905 ± 0.015 |
| **DINOv2** | **0.994 ± 0.003** | **0.999 ± 0.001** | **0.952 ± 0.008** |

---

## Publication Materials

### Generate All Figures

```bash
python scripts/generate_figures.py --all --output-dir figures/
```

**Outputs**:
- `precision_recall_comparison.png` - PR curves for all models
- `reliability_*.png` - Calibration diagrams per model
- `threshold_curves_*.png` - P/R/F1 vs threshold with operating points
- `fold_boxplots_*.png` - Cross-validation variance
- `confusion_matrix_*.png` - At optimal threshold
- `model_comparison_*.png` - Bar charts with bootstrap CIs

### Generate Dataset Card

```bash
python scripts/generate_dataset_card.py \
    --labels-file data/labels.json \
    --splits-file data/splits.json \
    --output docs/DATASET_CARD.md
```

### Generate Supplementary Tables

```bash
python scripts/generate_supplementary.py \
    --results-dir results/ \
    --configs-dir configs/ \
    --output-dir docs/supplementary/
```

**Outputs** (Markdown + LaTeX):
- `fold_metrics.md/.tex` - Per-fold AUROC, AUPRC, F1
- `hyperparameters.md/.tex` - Consolidated config comparison
- `mcnemar_test.md/.tex` - Pairwise statistical tests
- `operating_points.md/.tex` - Precision at 90%, 95%, 99% recall

### Generate Model Cards

```bash
python scripts/generate_model_card.py \
    --model model_c \
    --checkpoint results/model_c/fold_0/best.ckpt \
    --output docs/MODEL_CARD_C.md
```

---

## Reproducibility

### Lock Dependencies

```bash
python scripts/lock_requirements.py --output requirements-lock.txt
```

### Dataset Versioning

```python
from blue_frogs.data.versioning import compute_dataset_hash, save_dataset_hash

hash_info = compute_dataset_hash(Path("data/"))
save_dataset_hash(hash_info, Path("data/dataset_hashes.json"))
```

### Verify Dataset Integrity

```python
from blue_frogs.data.versioning import verify_against_file

result = verify_against_file(Path("data/"), Path("data/dataset_hashes.json"))
assert result["valid"], f"Dataset mismatch: {result['file_mismatches']}"
```

### Full Reproduction

```bash
# 1. Clone and setup
git clone https://github.com/dangause/blue-frogs.git
cd blue-frogs
pip install -r requirements-lock.txt

# 2. Verify data integrity
python -c "from blue_frogs.data.versioning import verify_against_file; print(verify_against_file('data/', 'data/dataset_hashes.json'))"

# 3. Train (on GPU node)
bash scripts/spark/train_all.sh

# 4. Calibrate
python scripts/calibrate.py --results-dir results/

# 5. Aggregate and compare
python scripts/aggregate_results.py --results-dir results/
python scripts/compare_models.py --results-dir results/

# 6. Generate publication materials
python scripts/generate_figures.py --all
python scripts/generate_supplementary.py
```

---

## Appendix: File Reference

| Script | Purpose |
|--------|---------|
| `train.py` | Unified training for all models |
| `calibrate.py` | Post-hoc temperature scaling |
| `aggregate_results.py` | Combine fold results + bootstrap CIs |
| `compare_models.py` | McNemar's test for model comparison |
| `stream_inference.py` | Streaming inference over iNaturalist |
| `generate_figures.py` | Publication figures |
| `generate_dataset_card.py` | Dataset documentation |
| `generate_supplementary.py` | LaTeX/Markdown tables |
| `generate_model_card.py` | Model documentation |
| `lock_requirements.py` | Pin dependencies |

| Module | Purpose |
|--------|---------|
| `data/dataset.py` | PyTorch datasets |
| `data/curation.py` | Negative sampling tiers |
| `data/versioning.py` | Dataset hashing |
| `models/model_a.py` | EfficientNetV2 classifier |
| `models/model_b_classifier.py` | CNN-LAB fusion |
| `models/model_c.py` | DINOv2 classifier |
| `evaluation/metrics.py` | AUPRC, threshold finding |
| `evaluation/comparison.py` | McNemar's test |
| `inference/stream_scorer.py` | Streaming inference |
| `figures/paper_figures.py` | Main figures |
| `figures/supplementary_figures.py` | Supplementary figures |
