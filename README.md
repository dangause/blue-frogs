# Blue Frogs: Axanthism Classifier

Computer vision pipeline for detecting **axanthism** (blue coloration caused by loss of yellow xanthophore pigments) in frogs from iNaturalist images. Built to extend the [Womack et al. Blue Frogs Project](https://github.com/mcwomack/bluefrogs) dataset (~372 confirmed axanthic observations across 36 species) to automated, full-scale detection across the entire iNaturalist frog corpus.

## Four Model Architectures

| Model | Architecture | Approach |
|-------|-------------|----------|
| **A** | EfficientNetV2-S | Standard image classifier trained from scratch |
| **B** | YOLOv8 + CNN-LAB fusion | Detects the frog first, then classifies using image + color features |
| **C** | DINOv2 (86M params) | Pre-trained vision model fine-tuned for axanthism -- **best performer** |
| **D** | DINOv3 (300M params) | Larger pre-trained model; did not improve over Model C |

## Project Structure

```
blue-frogs/
├── src/blue_frogs/
│   ├── config.py              # Central paths & constants
│   ├── models/                # Model A, B, C, D implementations
│   ├── data/                  # Dataset, downloader, iNat API, splits
│   ├── evaluation/            # Metrics, model comparison, Grad-CAM
│   ├── inference/             # Batch + streaming scoring pipelines
│   └── figures/               # Publication figure generation
├── scripts/                   # CLI entry points
│   ├── train.py               # Unified training (all models)
│   ├── calibrate.py           # Post-hoc temperature scaling
│   ├── preprocess_crops.py    # YOLO crop + LAB pre-computation (Model B)
│   ├── stream_inference.py    # Streaming inference over iNat corpus
│   ├── smoke_test.py          # End-to-end integration test
│   ├── run_inference.py       # Batch inference on full corpus
│   ├── download_images.py     # Image download from iNaturalist
│   ├── compare_models.py      # Statistical model comparison
│   ├── generate_figures.py    # PR curves, LAB distributions
│   └── generate_gallery.py   # HTML gallery of flagged images
├── configs/                   # YAML hyperparameter configs
│   ├── model_a.yaml
│   ├── model_b.yaml
│   ├── model_c.yaml
│   ├── model_d.yaml
│   └── smoke_test.yaml
├── tests/                     # 58 unit/integration tests
├── docs/                      # Design documents & findings
├── Dockerfile                 # Multi-stage CPU build for smoke tests
├── pyproject.toml             # Package metadata & dependencies
└── environment.yml            # Conda environment (CUDA 12.1)
```

## Setup

### Conda (recommended for GPU training)

```bash
conda env create -f environment.yml
conda activate blue-frogs
```

### pip

```bash
pip install -e ".[dev]"

# Optional extras
pip install -e ".[bioclip]"   # BioCLIP backbone for Model C
pip install -e ".[yolo]"      # YOLOv8 detector for Model B
```

### uv (fast alternative)

```bash
uv sync --extra dev
uv sync --extra bioclip   # BioCLIP backbone for Model C
uv sync --extra yolo       # YOLOv8 detector for Model B
```

**Requirements:** Python 3.11+, PyTorch 2.1+

## Quick Start

### Run the smoke test (Docker, no GPU needed)

```bash
docker build -t blue-frogs-smoke .
docker run -v $(pwd)/data:/app/data blue-frogs-smoke
```

This downloads a subset of data, trains Model A for 3 epochs, and validates the full pipeline end-to-end.

### Train a model

```bash
# Model A (image-only CNN)
python scripts/train.py --config configs/model_a.yaml --model model_a \
  --data-dir data/ --labels-file data/labels.json --splits-file data/splits.json

# Model B (requires YOLO preprocessing first)
python scripts/preprocess_crops.py --data-dir data/ --labels-file data/labels.json
python scripts/train.py --config configs/model_b.yaml --model model_b \
  --data-dir data/ --labels-file data/labels.json --splits-file data/splits.json \
  --crop-dir data/cropped_images --lab-features data/crop_lab_features.npy \
  --lab-index data/crop_lab_index.json
```

### Run streaming inference

```bash
# Single model
python scripts/stream_inference.py \
  --models model_b \
  --checkpoints models/model_b/fold_0/finetune/best-val/auprc=0.9766.ckpt \
  --detector-checkpoint models/model_b_detector/frog_detector.pt \
  --obs-per-batch 200 --max-batches 5

# Multi-model with calibration
python scripts/stream_inference.py \
  --models model_c model_d \
  --checkpoints models/model_c/fold_0/best.ckpt models/model_d/fold_0/best.ckpt \
  --calibration-file results/calibration.json \
  --obs-per-batch 500 --save-flagged
```

### GPU training (Docker, full pipeline)

```bash
# Build the GPU image
docker build -f Dockerfile.gpu -t blue-frogs-gpu .

# Train all 4 models (5 folds each) + aggregate results
bash scripts/docker/train_all.sh

# Run streaming inference on the full iNat corpus
bash scripts/docker/run_inference.sh
```

Override defaults with environment variables:

```bash
DATA_DIR=/data/frogs RESULTS_DIR=/results GPU_ID=0 bash scripts/docker/train_all.sh
```

### Download images from iNaturalist

```bash
python scripts/download_images.py --data-dir data
```

## Data Pipeline

1. **Labels** from Womack et al. CSV (~372 confirmed axanthic observations)
2. **Metadata** fetched from iNaturalist API (rate-limited)
3. **Images** downloaded in parallel with caching
4. **Negatives** curated in three tiers:
   - **Hard** (~1,000): naturally blue species (e.g., *Dendrobates*)
   - **Medium** (~2,000): same species as positives, normal coloration
   - **Easy** (~7,000): random stratified across families
5. **Splits**: stratified 5-fold CV by observation (not photo) + 20% held-out test set

Augmentations include standard transforms with **hue constrained to +/-5 degrees** to preserve the color-based signal that defines axanthism.

## Evaluation

Primary metrics chosen for extreme class imbalance (~0.09% prevalence in the wild):

- **AUPRC** (primary) - overall detection quality across all confidence thresholds (higher = better)
- **AUROC** - similar measure, less sensitive to class imbalance
- **F1** - balance between catching axanthic frogs and not flagging normal ones (higher = better)
- **Precision @ 95% recall** / **Recall @ 95% precision** - operational trade-off metrics

All metrics include bootstrap confidence intervals (1000 resamples). Model comparison uses McNemar's test for statistical significance.

## Tests

```bash
pytest                    # Run all 58 tests
pytest tests/ --cov       # With coverage
ruff check src/ tests/    # Lint
```

## Results

### Model A (EfficientNetV2-S)

| Metric | Value |
|--------|-------|
| AUROC | 0.9894 |
| AUPRC | 0.9598 |
| F1 | 0.904 |

Smoke test: 652 images, 3 epochs, single fold, CPU.

### Model B (YOLO + CNN-LAB Fusion)

| Metric | Value |
|--------|-------|
| AUROC | 0.9833 |
| AUPRC | 0.9473 |
| F1 | 0.892 |

Two-stage training (linear probe + fine-tune), fold 0, YOLO-cropped images with pre-computed LAB features.

### Model C (DINOv2-Base) -- Best Overall

| Metric | Value |
|--------|-------|
| AUROC | 0.999 +/- 0.001 |
| AUPRC | 0.994 +/- 0.003 |
| F1 | 0.952 +/- 0.008 |

5-fold cross-validation. Recommended for production screening of iNaturalist images.

### Model D (DINOv3 ViT-L/16)

| Metric | Value |
|--------|-------|
| AUROC | 0.998 +/- 0.001 |
| AUPRC | 0.992 +/- 0.002 |
| F1 | 0.948 +/- 0.006 |

5-fold cross-validation. Larger model (300M vs 86M parameters) that did not improve over Model C. See [findings writeup](docs/findings-model-d.md) for why bigger wasn't better and recommendations for next steps.

## Design Documents

Detailed architecture and implementation plans are in `docs/plans/`:

- `axanthism-classifier-design.md` - Scientific context, model specifications, evaluation protocol
- `axanthism-classifier-plan.md` - Implementation task breakdown
- `smoke-test-design.md` - Integration test design for local validation
- `smoke-test-plan.md` - Smoke test implementation checklist
- `data-preparation-and-hpc-runbook-design.md` - Data prep and HPC deployment

## License

TBD
