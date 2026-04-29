# HPC Training Infrastructure Design

**Date:** 2026-03-01
**Status:** Approved
**Scope:** Complete train.py, GPU Dockerfile, SLURM job scripts, fold aggregation, inference updates

## Context

Three model architectures (A, B, C) have passed smoke tests on CPU. The next step is
full-scale 5-fold cross-validation training on the CAS ibss-genomics SLURM cluster using
Apptainer containers with GPU support.

### Cluster Environment

- **Scheduler:** SLURM 23.11.4 on ibss-genomics (head node)
- **GPU node:** alice (currently being set up — nvidia-smi issue pending resolution)
- **Other nodes:** flor, rosalindf, tdobz (CPU-only)
- **Storage:** NFS /home (1.4 PB)
- **Containers:** Apptainer — build Docker locally, push to Docker Hub, pull as SIF on cluster
- **Cache:** `/var/tmp/apptainer` (local disk, nightly cleanup — pre-pull SIF to ~/containers/)
- **No default resource limits** — jobs run until they terminate

### Training Matrix

| Model | Architecture | Folds | Batch Size | Est. Memory | Special |
|-------|-------------|-------|------------|-------------|---------|
| A | EfficientNetV2-S | 5 | 32 | ~16 GB | Standard |
| B | YOLO + CNN-LAB Fusion | 5 | 32 | ~16 GB | YOLO pre-detection step |
| C | DINOv2 ViT-B/14 | 5 | 16 | ~32 GB | Two-stage (linear probe + fine-tune) |

## Approach: SLURM Array Jobs Per Model

One sbatch file per model using `--array=0-4` for the 5 folds. A wrapper script submits
all three with SLURM dependency chains for aggregation.

**Rationale:** Models have genuinely different resource needs and training patterns
(especially Model C's two-stage approach). Per-model sbatch files are the cleanest fit.
Standard HPC pattern — anyone familiar with SLURM understands it immediately.

## Components

### 1. Complete `train.py`

Current state: `main()` is a stub (lines 109-112 are placeholder comments).

Changes needed:

- **Add Model B** to `MODEL_CLASSES` dict (currently only has A and C)
- **Load metadata** from a JSON labels file (`--labels-file` argument):
  - Format: list of `{observation_id, photo_id, label, image_path}`
- **Load splits** using `get_fold_indices()` from `splits.py`
- **Build model kwargs** from config via a helper function per model:
  - Model A: `backbone`, `pretrained`, `dropout`, `loss_type`, `pos_weight`, `learning_rate`, etc.
  - Model B: `backbone`, `pretrained`, `color_feature_dim`, `fusion_hidden`, `dropout`, plus loss/optim
  - Model C: `backbone`, `model_size`, `hidden_dim`, `dropout`, plus loss/optim
- **Handle `--fold N`** for single-fold HPC execution (skip fold loop)
- **Handle Model C two-stage training:**
  1. Linear probe: freeze backbone, train for `linear_probe.max_epochs`
  2. Fine-tune: unfreeze backbone, load linear probe checkpoint, train for remaining epochs
     with `backbone_lr_factor` applied to backbone parameters
- **Add `--precision` flag** (default `"16-mixed"` for GPU, `"32"` for CPU)
- **Add `--wandb-offline` flag** (for clusters without internet — logs sync later)
- **Save fold summary** to `{output_dir}/{model}/fold_{N}/results.json`:
  - Best checkpoint path, best val/auprc, epochs trained

### 2. GPU Dockerfile (`Dockerfile.gpu`)

Separate from existing CPU smoke-test Dockerfile.

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# Install system deps, project with all extras
# No ENTRYPOINT — Apptainer calls python directly
```

Key decisions:
- **Base:** `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` (matches cluster CUDA expectations)
- **Install all extras:** `.[bioclip,yolo]` — single image for all models
- **No ENTRYPOINT** — Apptainer overrides entry via `apptainer exec ... python script.py`
- **Multi-stage build** to minimize image size (~5-6 GB)
- **Pre-pull on cluster:** `apptainer pull ~/containers/blue-frogs-gpu.sif docker://dangause/blue-frogs-gpu:latest`

### 3. SLURM Job Scripts

```
scripts/slurm/
├── submit_all.sh              # Orchestrator: submits A, B, C arrays + aggregation
├── train_model_a.sbatch       # EfficientNetV2-S, 5 folds
├── train_model_b.sbatch       # YOLO+Fusion, 5 folds
├── train_model_c.sbatch       # DINOv2 two-stage, 5 folds
└── aggregate_results.sbatch   # Post-training comparison
```

**Per-model sbatch template:**
```bash
#!/bin/bash
#SBATCH --job-name=bf-model-a
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/model_a_fold%a_%j.log
#SBATCH --nodelist=alice

FOLD=$SLURM_ARRAY_TASK_ID
SIF=~/containers/blue-frogs-gpu.sif
DATA=$HOME/blue-frogs/data
OUT=$HOME/blue-frogs/results

apptainer exec --nv \
  --bind $DATA:/app/data,$OUT:/app/results \
  $SIF python scripts/train.py \
    --model model_a \
    --config configs/model_a.yaml \
    --data-dir /app/data \
    --labels-file /app/data/labels.json \
    --splits-file /app/data/splits.json \
    --output-dir /app/results \
    --fold $FOLD \
    --precision 16-mixed \
    --wandb-offline
```

**Model-specific resource overrides:**
- Model A: `--mem=32G`
- Model B: `--mem=32G`
- Model C: `--mem=48G` (DINOv2 ViT-B/14 with fine-tuning is larger)

**`submit_all.sh` dependency chain:**
```bash
#!/bin/bash
set -euo pipefail
mkdir -p logs

JOB_A=$(sbatch --parsable scripts/slurm/train_model_a.sbatch)
JOB_B=$(sbatch --parsable scripts/slurm/train_model_b.sbatch)
JOB_C=$(sbatch --parsable scripts/slurm/train_model_c.sbatch)

echo "Submitted: A=$JOB_A  B=$JOB_B  C=$JOB_C"

AGG=$(sbatch --parsable \
  --dependency=afterok:$JOB_A:$JOB_B:$JOB_C \
  scripts/slurm/aggregate_results.sbatch)

echo "Aggregation: $AGG (runs after all training completes)"
```

### 4. Fold Aggregation Script (`scripts/aggregate_folds.py`)

New script that runs after all training completes:

1. **Collect** best checkpoints from `results/{model}/fold_{N}/` directories
2. **Run test-set evaluation** per fold checkpoint on held-out test set
3. **Compute ensemble predictions** (average probabilities across 5 folds)
4. **Generate comparison table** using existing `evaluation/metrics.py`:
   - AUPRC (primary), AUROC, F1, precision, recall — with bootstrap 95% CIs
5. **Run McNemar's pairwise tests** using existing `evaluation/comparison.py`
6. **Save outputs:**
   - `results/comparison_summary.json` — machine-readable
   - `results/comparison_summary.md` — human-readable markdown table

### 5. Update `run_inference.py`

Currently hardcoded to `EfficientNetClassifier` (Model A only).

Changes:
- Add `--model` argument (model_a, model_b, model_c)
- Model B support: run YOLO detection first, then classify crops with LAB features
- Model C support: load DINOv2/BioCLIP checkpoint
- Add `--ensemble` flag: average predictions across fold checkpoints (requires `--checkpoint-dir`)

### 6. Mixed Precision & Performance

Applied in `train.py` and `train_fold()`:
- `precision="16-mixed"` on Trainer when GPU detected
- `torch.set_float32_matmul_precision("medium")` for Ampere+ GPUs (TF32)
- DataLoader `num_workers` configurable via config (default 4)
- `pin_memory=True` (already present)

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `scripts/train.py` | Modify | Complete main(), add Model B, two-stage Model C, precision/wandb flags |
| `Dockerfile.gpu` | Create | CUDA-based image for all 3 models, no entrypoint |
| `scripts/slurm/submit_all.sh` | Create | Orchestrator that submits all jobs with dependencies |
| `scripts/slurm/train_model_a.sbatch` | Create | SLURM array job for Model A folds |
| `scripts/slurm/train_model_b.sbatch` | Create | SLURM array job for Model B folds |
| `scripts/slurm/train_model_c.sbatch` | Create | SLURM array job for Model C folds |
| `scripts/slurm/aggregate_results.sbatch` | Create | Post-training aggregation job |
| `scripts/aggregate_folds.py` | Create | Fold collection, ensemble eval, comparison |
| `scripts/run_inference.py` | Modify | Multi-model support, ensemble mode |
| `configs/model_a.yaml` | No change | Already complete |
| `configs/model_b.yaml` | No change | Already complete |
| `configs/model_c.yaml` | No change | Already complete |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| alice GPU node not ready | SLURM scripts work as-is once GPUs available; can CPU-test the full pipeline on other nodes |
| OOM on GPU | Model C already tested with batch_size=2 on CPU; GPU has more memory, start with config defaults and adjust |
| Apptainer SIF cache eviction | Pre-pull to ~/containers/ per CAS docs |
| WandB unavailable on cluster | `--wandb-offline` flag stores logs locally, sync after training |
| YOLO weights download in container | Pre-download yolov8m.pt into data dir, bind-mount it |
