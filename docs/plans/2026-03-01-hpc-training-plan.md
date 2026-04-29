# HPC Training Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the training pipeline so all three models can be trained via SLURM array jobs on the CAS ibss-genomics HPC cluster using Apptainer containers.

**Architecture:** Per-model SLURM array sbatch scripts (`--array=0-4`) submit 5-fold CV jobs that each call a completed `train.py` inside a GPU Docker image via `apptainer exec --nv`. A wrapper script chains aggregation via `--dependency=afterok`.

**Tech Stack:** PyTorch Lightning, SLURM, Apptainer, Docker (CUDA 11.8), WandB (offline mode)

---

### Task 1: Add Model B to train.py and build model kwargs helper

**Files:**
- Modify: `scripts/train.py:17-26` (imports and MODEL_CLASSES)
- Test: `tests/test_train.py` (new file)

**Step 1: Write failing test for build_model_kwargs**

Create `tests/test_train.py`:

```python
"""Tests for the unified training script."""

import pytest
import yaml


def test_build_model_kwargs_model_a():
    """build_model_kwargs extracts the right constructor args for Model A."""
    from scripts.train import build_model_kwargs

    config = {
        "model": {
            "backbone": "tf_efficientnetv2_s",
            "pretrained": True,
            "dropout": 0.3,
            "freeze_backbone": False,
        },
        "training": {
            "loss_type": "weighted_bce",
            "pos_weight": 20.0,
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
    }
    kwargs = build_model_kwargs("model_a", config)
    assert kwargs["backbone"] == "tf_efficientnetv2_s"
    assert kwargs["pretrained"] is True
    assert kwargs["dropout"] == 0.3
    assert kwargs["loss_type"] == "weighted_bce"
    assert kwargs["pos_weight"] == 20.0
    assert kwargs["optimizer"] == "sgd"
    assert kwargs["learning_rate"] == 0.001


def test_build_model_kwargs_model_b():
    """build_model_kwargs extracts fusion-specific args for Model B."""
    from scripts.train import build_model_kwargs

    config = {
        "classifier": {
            "backbone": "tf_efficientnetv2_s",
            "pretrained": True,
            "color_feature_dim": 30,
            "fusion_hidden": 256,
            "dropout": 0.3,
        },
        "training": {
            "loss_type": "weighted_bce",
            "pos_weight": 20.0,
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
    }
    kwargs = build_model_kwargs("model_b", config)
    assert kwargs["color_feature_dim"] == 30
    assert kwargs["fusion_hidden"] == 256
    assert kwargs["backbone"] == "tf_efficientnetv2_s"


def test_build_model_kwargs_model_c():
    """build_model_kwargs extracts foundation model args for Model C."""
    from scripts.train import build_model_kwargs

    config = {
        "model": {
            "backbone": "dinov2",
            "model_size": "base",
            "hidden_dim": 256,
            "dropout": 0.3,
        },
        "training": {
            "loss_type": "weighted_bce",
            "pos_weight": 20.0,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
        },
    }
    kwargs = build_model_kwargs("model_c", config)
    assert kwargs["backbone"] == "dinov2"
    assert kwargs["model_size"] == "base"
    assert kwargs["hidden_dim"] == 256
    assert kwargs["optimizer"] == "adamw"


def test_model_classes_contains_all_three():
    """MODEL_CLASSES dict has entries for all three models."""
    from scripts.train import MODEL_CLASSES

    assert "model_a" in MODEL_CLASSES
    assert "model_b" in MODEL_CLASSES
    assert "model_c" in MODEL_CLASSES
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `ImportError` for `build_model_kwargs` and missing `model_b`

**Step 3: Implement build_model_kwargs and add Model B import**

In `scripts/train.py`, add the FusionClassifier import and the helper function:

```python
# At line 18, add:
from blue_frogs.models.model_b_classifier import FusionClassifier

# Update MODEL_CLASSES at line 23:
MODEL_CLASSES = {
    "model_a": EfficientNetClassifier,
    "model_b": FusionClassifier,
    "model_c": FoundationModelClassifier,
}

# Add after MODEL_CLASSES, before train_fold:
def build_model_kwargs(model_name: str, config: dict) -> dict:
    """Extract model constructor kwargs from config.

    Each model has different config structure, so this maps config sections
    to the kwargs expected by each model class constructor.
    """
    training = config["training"]
    common_kwargs = {
        "loss_type": training["loss_type"],
        "pos_weight": training["pos_weight"],
        "optimizer": training["optimizer"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training.get("weight_decay", 0.0),
    }

    if model_name == "model_a":
        model_cfg = config["model"]
        return {
            **common_kwargs,
            "backbone": model_cfg["backbone"],
            "pretrained": model_cfg["pretrained"],
            "dropout": model_cfg["dropout"],
            "freeze_backbone": model_cfg.get("freeze_backbone", False),
        }
    elif model_name == "model_b":
        cls_cfg = config["classifier"]
        return {
            **common_kwargs,
            "backbone": cls_cfg["backbone"],
            "pretrained": cls_cfg["pretrained"],
            "color_feature_dim": cls_cfg["color_feature_dim"],
            "fusion_hidden": cls_cfg["fusion_hidden"],
            "dropout": cls_cfg["dropout"],
        }
    elif model_name == "model_c":
        model_cfg = config["model"]
        return {
            **common_kwargs,
            "backbone": model_cfg["backbone"],
            "model_size": model_cfg["model_size"],
            "hidden_dim": model_cfg["hidden_dim"],
            "dropout": model_cfg["dropout"],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add scripts/train.py tests/test_train.py
git commit -m "Add Model B to train.py and build_model_kwargs helper"
```

---

### Task 2: Complete train.py main() — data loading, splits, fold loop

**Files:**
- Modify: `scripts/train.py:91-116` (main function)
- Test: `tests/test_train.py` (add integration tests)

**Step 1: Write failing test for main() data loading path**

Add to `tests/test_train.py`:

```python
import json
import numpy as np
from pathlib import Path
from PIL import Image


@pytest.fixture
def synthetic_training_data(tmp_path):
    """Create a minimal dataset with labels, images, and splits for testing main()."""
    # Create images
    image_dir = tmp_path / "raw_images"
    image_dir.mkdir()
    metadata = []
    n_positive = 10
    n_negative = 40

    for i in range(n_positive + n_negative):
        label = 1 if i < n_positive else 0
        obs_id = 1000 + i
        photo_id = 2000 + i
        obs_dir = image_dir / str(obs_id)
        obs_dir.mkdir(exist_ok=True)
        img = Image.new("RGB", (100, 100), color=(50, 100, 200) if label else (0, 128, 0))
        photo_path = f"{obs_id}/{photo_id}.jpg"
        img.save(image_dir / photo_path)
        metadata.append({
            "observation_id": obs_id,
            "photo_id": photo_id,
            "photo_path": photo_path,
            "label": label,
        })

    labels_file = tmp_path / "labels.json"
    labels_file.write_text(json.dumps(metadata))

    # Create splits file (pre-computed train/test split indices)
    rng = np.random.RandomState(42)
    indices = list(range(len(metadata)))
    rng.shuffle(indices)
    test_size = int(0.2 * len(metadata))
    splits = {
        "test_indices": indices[:test_size],
        "train_indices": indices[test_size:],
    }
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps(splits))

    return tmp_path, labels_file, splits_file, image_dir


def test_load_training_data(synthetic_training_data):
    """load_training_data returns train_df and test_df with expected columns."""
    from scripts.train import load_training_data

    data_dir, labels_file, splits_file, image_dir = synthetic_training_data
    train_meta, test_meta = load_training_data(labels_file, splits_file)

    assert len(train_meta) > 0
    assert len(test_meta) > 0
    assert len(train_meta) + len(test_meta) == 50
    assert all("label" in m for m in train_meta)
    assert all("photo_path" in m for m in train_meta)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_load_training_data -v`
Expected: FAIL — `ImportError` for `load_training_data`

**Step 3: Implement load_training_data and complete main()**

Replace the stub `main()` in `scripts/train.py` (lines 91-116) with:

```python
def load_training_data(
    labels_file: Path, splits_file: Path
) -> tuple[list[dict], list[dict]]:
    """Load metadata and split into train/test sets.

    labels_file: JSON list of {observation_id, photo_id, photo_path, label}
    splits_file: JSON with {train_indices: [...], test_indices: [...]}
    """
    import json

    with open(labels_file) as f:
        all_metadata = json.load(f)
    with open(splits_file) as f:
        splits = json.load(f)

    train_meta = [all_metadata[i] for i in splits["train_indices"]]
    test_meta = [all_metadata[i] for i in splits["test_indices"]]
    return train_meta, test_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--fold", type=int, default=None, help="Train single fold (for HPC)")
    parser.add_argument(
        "--precision", type=str, default="16-mixed",
        help="Training precision: 16-mixed, bf16-mixed, or 32",
    )
    parser.add_argument(
        "--wandb-offline", action="store_true",
        help="Run WandB in offline mode (sync logs later)",
    )
    args = parser.parse_args()

    if args.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model_name"] = args.model
    model_class = MODEL_CLASSES[args.model]
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_meta, test_meta = load_training_data(args.labels_file, args.splits_file)
    logger.info(f"Training {args.model} with config {args.config}")
    logger.info(f"Train: {len(train_meta)}, Test: {len(test_meta)}")
    logger.info(f"Output: {output_dir}")

    # Determine folds to train
    import pandas as pd
    train_df = pd.DataFrame(train_meta)
    n_folds = config.get("data", {}).get("n_folds", 5)
    seed = config.get("data", {}).get("seed", RANDOM_SEED)
    folds = get_fold_indices(train_df, n_folds=n_folds, seed=seed)

    if args.fold is not None:
        fold_list = [args.fold]
    else:
        fold_list = list(range(n_folds))

    image_dir = args.data_dir / "raw_images"
    model_kwargs = build_model_kwargs(args.model, config)
    results = {}

    for fold_idx in fold_list:
        logger.info(f"--- Fold {fold_idx}/{n_folds - 1} ---")
        train_idx, val_idx = folds[fold_idx]

        fold_train_meta = [train_meta[i] for i in train_idx]
        fold_val_meta = [train_meta[i] for i in val_idx]

        train_dataset = FrogDataset(fold_train_meta, image_dir, transform=get_train_transforms())
        val_dataset = FrogDataset(fold_val_meta, image_dir, transform=get_val_transforms())

        if args.model == "model_c" and "linear_probe" in config:
            best_path = train_fold_two_stage(
                model_class, model_kwargs, train_dataset, val_dataset,
                config, fold_idx, output_dir, args.precision,
            )
        else:
            best_path = train_fold(
                model_class, model_kwargs, train_dataset, val_dataset,
                config, fold_idx, output_dir, args.precision,
            )

        results[fold_idx] = {"best_checkpoint": best_path}
        logger.info(f"Fold {fold_idx} best checkpoint: {best_path}")

    # Save fold results summary
    import json
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Training summary saved to {summary_path}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py::test_load_training_data -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/train.py tests/test_train.py
git commit -m "Complete train.py main() with data loading, splits, and fold loop"
```

---

### Task 3: Add precision and two-stage training to train_fold

**Files:**
- Modify: `scripts/train.py:29-88` (train_fold function, add train_fold_two_stage)
- Test: `tests/test_train.py` (add tests)

**Step 1: Write failing test for train_fold precision parameter**

Add to `tests/test_train.py`:

```python
import inspect


def test_train_fold_accepts_precision():
    """train_fold accepts a precision parameter."""
    from scripts.train import train_fold
    sig = inspect.signature(train_fold)
    assert "precision" in sig.parameters


def test_train_fold_two_stage_exists():
    """train_fold_two_stage function exists and has expected parameters."""
    from scripts.train import train_fold_two_stage
    sig = inspect.signature(train_fold_two_stage)
    assert "precision" in sig.parameters
    assert "config" in sig.parameters
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py::test_train_fold_accepts_precision tests/test_train.py::test_train_fold_two_stage_exists -v`
Expected: FAIL

**Step 3: Update train_fold and add train_fold_two_stage**

Update `train_fold` signature to include precision, and add `train_fold_two_stage`:

```python
def train_fold(
    model_class,
    model_kwargs: dict,
    train_dataset: FrogDataset,
    val_dataset: FrogDataset,
    config: dict,
    fold: int,
    output_dir: Path,
    precision: str = "16-mixed",
):
    """Train a single fold."""
    pl.seed_everything(RANDOM_SEED + fold)

    model = model_class(**model_kwargs)

    train_labels = [m["label"] for m in train_dataset.metadata]
    sampler = make_weighted_sampler(train_labels)

    num_workers = config["training"].get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=fold_dir,
        filename="best-{val/auprc:.4f}",
        monitor="val/auprc",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc",
        mode="max",
        patience=config["training"]["early_stopping_patience"],
    )
    wandb_logger = WandbLogger(
        project="blue-frogs",
        name=f"{config.get('model_name', 'model')}_fold{fold}",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        precision=precision,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=wandb_logger,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    return checkpoint_cb.best_model_path


def train_fold_two_stage(
    model_class,
    model_kwargs: dict,
    train_dataset: FrogDataset,
    val_dataset: FrogDataset,
    config: dict,
    fold: int,
    output_dir: Path,
    precision: str = "16-mixed",
):
    """Train Model C in two stages: linear probe (frozen) then fine-tune (unfrozen).

    Follows the same pattern as smoke_test_c.py:step_train_two_stage but integrated
    into the unified training pipeline with WandB logging and fold support.
    """
    pl.seed_everything(RANDOM_SEED + fold)

    num_workers = config["training"].get("num_workers", 4)
    batch_size = config["training"]["batch_size"]
    fold_dir = output_dir / f"fold_{fold}"

    # --- Stage 1: Linear probe (frozen backbone) ---
    logger.info(f"Fold {fold} — Stage 1: Linear Probe (frozen backbone)")
    probe_cfg = config["linear_probe"]

    probe_kwargs = {**model_kwargs, "freeze_backbone": True, "learning_rate": probe_cfg["learning_rate"]}
    model = model_class(**probe_kwargs)

    train_labels = [m["label"] for m in train_dataset.metadata]
    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    probe_dir = fold_dir / "linear_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=probe_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )
    wandb_logger = WandbLogger(
        project="blue-frogs",
        name=f"{config.get('model_name', 'model')}_fold{fold}_probe",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=probe_cfg["max_epochs"], accelerator="auto", precision=precision,
        callbacks=[checkpoint_cb], logger=wandb_logger, deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    probe_ckpt = checkpoint_cb.best_model_path
    logger.info(f"Linear probe complete. Best: {probe_ckpt}")

    # --- Stage 2: Fine-tune (unfrozen backbone, differential LR) ---
    logger.info(f"Fold {fold} — Stage 2: Fine-tune (unfrozen backbone)")
    ft_cfg = config["finetune"]

    probe_model = model_class.load_from_checkpoint(probe_ckpt)
    ft_kwargs = {**model_kwargs, "freeze_backbone": False}
    model = model_class(**ft_kwargs)
    model.head.load_state_dict(probe_model.head.state_dict())

    ft_batch_size = ft_cfg.get("batch_size", batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=ft_batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=ft_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    ft_dir = fold_dir / "finetune"
    ft_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ft_dir, filename="best-{val/auprc:.4f}",
        monitor="val/auprc", mode="max", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc", mode="max",
        patience=config["training"]["early_stopping_patience"],
    )
    wandb_logger = WandbLogger(
        project="blue-frogs",
        name=f"{config.get('model_name', 'model')}_fold{fold}_finetune",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=ft_cfg.get("max_epochs", config["training"]["max_epochs"]),
        accelerator="auto", precision=precision,
        callbacks=[checkpoint_cb, early_stop_cb], logger=wandb_logger,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Fine-tune complete. Best: {checkpoint_cb.best_model_path}")

    return checkpoint_cb.best_model_path
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: All PASSED

**Step 5: Run full test suite to verify no regressions**

Run: `pytest`
Expected: All 38+ tests PASS

**Step 6: Commit**

```bash
git add scripts/train.py tests/test_train.py
git commit -m "Add precision support and two-stage training to train_fold"
```

---

### Task 4: Create GPU Dockerfile

**Files:**
- Create: `Dockerfile.gpu`
- Test: Build the image locally

**Step 1: Write Dockerfile.gpu**

```dockerfile
# GPU training image for HPC (Apptainer-compatible)
# Build: docker build -f Dockerfile.gpu -t blue-frogs-gpu .
# Push:  docker push dangause/blue-frogs-gpu:latest
# Pull on HPC: apptainer pull ~/containers/blue-frogs-gpu.sif docker://dangause/blue-frogs-gpu:latest

# --- Build stage ---
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

# Install project with all extras (bioclip for Model C, yolo for Model B)
RUN pip install --no-cache-dir ".[bioclip,yolo]"

# --- Runtime stage ---
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

# Copy project files needed at runtime
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Data and results are bind-mounted at runtime
VOLUME ["/app/data", "/app/results"]

# No ENTRYPOINT — Apptainer calls python directly:
#   apptainer exec --nv blue-frogs-gpu.sif python scripts/train.py ...
```

**Step 2: Verify Dockerfile builds (optional — depends on having Docker + CUDA locally)**

Run: `docker build -f Dockerfile.gpu -t blue-frogs-gpu . --no-cache 2>&1 | tail -5`

If no local CUDA available, just verify syntax: `docker build -f Dockerfile.gpu -t blue-frogs-gpu . --target builder`

**Step 3: Commit**

```bash
git add Dockerfile.gpu
git commit -m "Add GPU Dockerfile for Apptainer-based HPC training"
```

---

### Task 5: Create SLURM job scripts for Model A

**Files:**
- Create: `scripts/slurm/train_model_a.sbatch`
- Create: `scripts/slurm/submit_all.sh`

**Step 1: Create directory**

```bash
mkdir -p scripts/slurm
```

**Step 2: Write train_model_a.sbatch**

```bash
#!/bin/bash
#SBATCH --job-name=bf-model-a
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/model_a_fold%a_%j.log
#SBATCH --error=logs/model_a_fold%a_%j.err
#SBATCH --nodelist=alice

# Model A: EfficientNetV2-S, 5-fold cross-validation
# Each array task trains one fold.

set -euo pipefail

FOLD=$SLURM_ARRAY_TASK_ID
SIF="${SIF:-$HOME/containers/blue-frogs-gpu.sif}"
DATA="${DATA_DIR:-$HOME/blue-frogs/data}"
OUT="${OUTPUT_DIR:-$HOME/blue-frogs/results}"
CONFIG="${CONFIG:-configs/model_a.yaml}"

echo "=== Model A | Fold $FOLD | $(date) ==="
echo "SIF: $SIF"
echo "Data: $DATA"
echo "Output: $OUT"

apptainer exec --nv \
    --bind "$DATA":/app/data,"$OUT":/app/results \
    "$SIF" python scripts/train.py \
        --model model_a \
        --config "$CONFIG" \
        --data-dir /app/data \
        --labels-file /app/data/labels.json \
        --splits-file /app/data/splits.json \
        --output-dir /app/results \
        --fold "$FOLD" \
        --precision 16-mixed \
        --wandb-offline

echo "=== Model A | Fold $FOLD complete | $(date) ==="
```

**Step 3: Write submit_all.sh**

```bash
#!/bin/bash
# Submit all model training jobs with dependency-chained aggregation.
#
# Usage: bash scripts/slurm/submit_all.sh
#
# Override defaults with environment variables:
#   SIF=~/containers/blue-frogs-gpu.sif DATA_DIR=~/data OUTPUT_DIR=~/results bash scripts/slurm/submit_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

echo "Submitting training jobs..."

JOB_A=$(sbatch --parsable "$SCRIPT_DIR/train_model_a.sbatch")
echo "  Model A: job $JOB_A (array 0-4)"

JOB_B=$(sbatch --parsable "$SCRIPT_DIR/train_model_b.sbatch")
echo "  Model B: job $JOB_B (array 0-4)"

JOB_C=$(sbatch --parsable "$SCRIPT_DIR/train_model_c.sbatch")
echo "  Model C: job $JOB_C (array 0-4)"

echo ""
echo "Submitting aggregation job (depends on all training)..."
AGG=$(sbatch --parsable \
    --dependency="afterok:$JOB_A:$JOB_B:$JOB_C" \
    "$SCRIPT_DIR/aggregate_results.sbatch")
echo "  Aggregation: job $AGG"

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $JOB_A $JOB_B $JOB_C $AGG"
```

**Step 4: Make scripts executable**

```bash
chmod +x scripts/slurm/submit_all.sh
```

**Step 5: Commit**

```bash
git add scripts/slurm/train_model_a.sbatch scripts/slurm/submit_all.sh
git commit -m "Add SLURM job script for Model A and submission orchestrator"
```

---

### Task 6: Create SLURM job scripts for Models B and C

**Files:**
- Create: `scripts/slurm/train_model_b.sbatch`
- Create: `scripts/slurm/train_model_c.sbatch`

**Step 1: Write train_model_b.sbatch**

Same structure as Model A but with `--model model_b` and `configs/model_b.yaml`:

```bash
#!/bin/bash
#SBATCH --job-name=bf-model-b
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/model_b_fold%a_%j.log
#SBATCH --error=logs/model_b_fold%a_%j.err
#SBATCH --nodelist=alice

# Model B: YOLO + CNN-LAB Fusion, 5-fold cross-validation

set -euo pipefail

FOLD=$SLURM_ARRAY_TASK_ID
SIF="${SIF:-$HOME/containers/blue-frogs-gpu.sif}"
DATA="${DATA_DIR:-$HOME/blue-frogs/data}"
OUT="${OUTPUT_DIR:-$HOME/blue-frogs/results}"
CONFIG="${CONFIG:-configs/model_b.yaml}"

echo "=== Model B | Fold $FOLD | $(date) ==="

apptainer exec --nv \
    --bind "$DATA":/app/data,"$OUT":/app/results \
    "$SIF" python scripts/train.py \
        --model model_b \
        --config "$CONFIG" \
        --data-dir /app/data \
        --labels-file /app/data/labels.json \
        --splits-file /app/data/splits.json \
        --output-dir /app/results \
        --fold "$FOLD" \
        --precision 16-mixed \
        --wandb-offline

echo "=== Model B | Fold $FOLD complete | $(date) ==="
```

**Step 2: Write train_model_c.sbatch**

Model C gets more memory (48G) for DINOv2 ViT-B/14 fine-tuning:

```bash
#!/bin/bash
#SBATCH --job-name=bf-model-c
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/model_c_fold%a_%j.log
#SBATCH --error=logs/model_c_fold%a_%j.err
#SBATCH --nodelist=alice

# Model C: DINOv2 two-stage (linear probe + fine-tune), 5-fold cross-validation
# Uses 48G memory — DINOv2 ViT-B/14 fine-tuning with full backbone unfrozen.

set -euo pipefail

FOLD=$SLURM_ARRAY_TASK_ID
SIF="${SIF:-$HOME/containers/blue-frogs-gpu.sif}"
DATA="${DATA_DIR:-$HOME/blue-frogs/data}"
OUT="${OUTPUT_DIR:-$HOME/blue-frogs/results}"
CONFIG="${CONFIG:-configs/model_c.yaml}"

echo "=== Model C | Fold $FOLD | $(date) ==="

apptainer exec --nv \
    --bind "$DATA":/app/data,"$OUT":/app/results \
    "$SIF" python scripts/train.py \
        --model model_c \
        --config "$CONFIG" \
        --data-dir /app/data \
        --labels-file /app/data/labels.json \
        --splits-file /app/data/splits.json \
        --output-dir /app/results \
        --fold "$FOLD" \
        --precision 16-mixed \
        --wandb-offline

echo "=== Model C | Fold $FOLD complete | $(date) ==="
```

**Step 3: Commit**

```bash
git add scripts/slurm/train_model_b.sbatch scripts/slurm/train_model_c.sbatch
git commit -m "Add SLURM job scripts for Models B and C"
```

---

### Task 7: Create fold aggregation script

**Files:**
- Create: `scripts/aggregate_folds.py`
- Create: `scripts/slurm/aggregate_results.sbatch`
- Test: `tests/test_aggregate.py`

**Step 1: Write failing test**

Create `tests/test_aggregate.py`:

```python
"""Tests for fold aggregation script."""

import json
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def mock_fold_results(tmp_path):
    """Create mock per-fold prediction files for aggregation testing."""
    results_dir = tmp_path / "model_a"
    n_samples = 50
    rng = np.random.RandomState(42)
    y_true = np.concatenate([np.ones(10), np.zeros(40)])

    for fold in range(3):  # 3 folds for speed
        fold_dir = results_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        # Simulate per-fold predictions
        y_score = rng.uniform(0, 1, size=n_samples)
        # Make positive samples score higher on average
        y_score[:10] += 0.3
        y_score = np.clip(y_score, 0, 1)
        preds = {
            "y_true": y_true.tolist(),
            "y_score": y_score.tolist(),
        }
        (fold_dir / "test_predictions.json").write_text(json.dumps(preds))

    return tmp_path


def test_aggregate_fold_predictions(mock_fold_results):
    """aggregate_fold_predictions averages predictions across folds."""
    from scripts.aggregate_folds import aggregate_fold_predictions

    y_true, y_score_ensemble = aggregate_fold_predictions(
        mock_fold_results / "model_a", n_folds=3
    )
    assert len(y_true) == 50
    assert len(y_score_ensemble) == 50
    assert 0.0 <= y_score_ensemble.min()
    assert y_score_ensemble.max() <= 1.0


def test_generate_comparison_table(mock_fold_results):
    """generate_comparison_table produces a markdown string."""
    from scripts.aggregate_folds import generate_comparison_table

    model_results = {
        "model_a": {
            "auprc": 0.85,
            "auprc_ci_lower": 0.80,
            "auprc_ci_upper": 0.90,
            "auroc": 0.92,
            "auroc_ci_lower": 0.88,
            "auroc_ci_upper": 0.96,
            "f1_optimal": 0.78,
        },
    }
    table = generate_comparison_table(model_results)
    assert "model_a" in table
    assert "AUPRC" in table
    assert "0.85" in table
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aggregate.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement aggregate_folds.py**

Create `scripts/aggregate_folds.py`:

```python
"""Aggregate per-fold results into ensemble predictions and comparison tables.

Run after all 5 folds complete for each model. Collects per-fold test predictions,
computes ensemble averages, and runs statistical comparison across models.

Usage:
    python scripts/aggregate_folds.py --results-dir results/ --output-dir results/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from blue_frogs.config import RESULTS_DIR
from blue_frogs.evaluation.comparison import compare_models, mcnemar_test
from blue_frogs.evaluation.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_fold_predictions(
    model_dir: Path, n_folds: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Average test predictions across folds for ensemble evaluation.

    Each fold directory should contain test_predictions.json with:
        {"y_true": [...], "y_score": [...]}

    Returns (y_true, y_score_ensemble) where y_score_ensemble is the
    mean of per-fold predicted probabilities.
    """
    all_scores = []
    y_true = None

    for fold in range(n_folds):
        pred_file = model_dir / f"fold_{fold}" / "test_predictions.json"
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_file}")

        with open(pred_file) as f:
            preds = json.load(f)

        scores = np.array(preds["y_score"])
        all_scores.append(scores)

        if y_true is None:
            y_true = np.array(preds["y_true"])

    y_score_ensemble = np.mean(all_scores, axis=0)
    return y_true, y_score_ensemble


def generate_comparison_table(model_results: dict) -> str:
    """Generate a markdown comparison table from model results."""
    lines = [
        "| Model | AUPRC | AUPRC 95% CI | AUROC | AUROC 95% CI | F1 |",
        "|-------|-------|-------------|-------|-------------|-----|",
    ]
    for name, metrics in sorted(model_results.items()):
        auprc = metrics["auprc"]
        auprc_ci = f"[{metrics['auprc_ci_lower']:.4f}, {metrics['auprc_ci_upper']:.4f}]"
        auroc = metrics["auroc"]
        auroc_ci = f"[{metrics['auroc_ci_lower']:.4f}, {metrics['auroc_ci_upper']:.4f}]"
        f1 = metrics["f1_optimal"]
        lines.append(f"| {name} | {auprc:.4f} | {auprc_ci} | {auroc:.4f} | {auroc_ci} | {f1:.4f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate fold results and compare models")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing per-model result subdirectories")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = ["model_a", "model_b", "model_c"]

    # Aggregate per-model ensemble predictions
    model_scores = {}
    y_true = None
    for name in model_names:
        model_dir = args.results_dir / name
        if not model_dir.exists():
            logger.warning(f"Skipping {name} — no results directory")
            continue
        try:
            yt, ys = aggregate_fold_predictions(model_dir, args.n_folds)
            model_scores[name] = ys
            if y_true is None:
                y_true = yt
        except FileNotFoundError as e:
            logger.warning(f"Skipping {name}: {e}")

    if not model_scores:
        logger.error("No model results found")
        return

    # Compare all models
    logger.info(f"Comparing {len(model_scores)} models...")
    results = compare_models(y_true, model_scores)

    # Print and save comparison table
    table = generate_comparison_table(results)
    logger.info(f"\n{table}")

    table_path = args.output_dir / "comparison_summary.md"
    table_path.write_text(f"# Model Comparison\n\n{table}\n")
    logger.info(f"Table saved to {table_path}")

    # Pairwise McNemar's tests
    names = sorted(model_scores.keys())
    mcnemar_results = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            thresh_a = results[name_a]["optimal_threshold"]
            thresh_b = results[name_b]["optimal_threshold"]
            preds_a = (model_scores[name_a] >= thresh_a).astype(int)
            preds_b = (model_scores[name_b] >= thresh_b).astype(int)
            mc = mcnemar_test(y_true, preds_a, preds_b)
            pair = f"{name_a}_vs_{name_b}"
            mcnemar_results[pair] = mc
            logger.info(f"McNemar {pair}: p={mc['p_value']:.4f}")

    # Save full results
    output = {"model_metrics": results, "mcnemar_tests": mcnemar_results}
    json_path = args.output_dir / "comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Write aggregate_results.sbatch**

```bash
#!/bin/bash
#SBATCH --job-name=bf-aggregate
#SBATCH --mem=16G
#SBATCH --output=logs/aggregate_%j.log
#SBATCH --error=logs/aggregate_%j.err

# Aggregation runs on CPU after all training jobs complete.
# Submitted with --dependency=afterok by submit_all.sh.

set -euo pipefail

SIF="${SIF:-$HOME/containers/blue-frogs-gpu.sif}"
OUT="${OUTPUT_DIR:-$HOME/blue-frogs/results}"

echo "=== Aggregation | $(date) ==="

apptainer exec \
    --bind "$OUT":/app/results \
    "$SIF" python scripts/aggregate_folds.py \
        --results-dir /app/results \
        --output-dir /app/results

echo "=== Aggregation complete | $(date) ==="
echo "Results at: $OUT/comparison_summary.md"
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_aggregate.py -v`
Expected: 2 PASSED

**Step 6: Run full test suite**

Run: `pytest`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add scripts/aggregate_folds.py scripts/slurm/aggregate_results.sbatch tests/test_aggregate.py
git commit -m "Add fold aggregation script and SLURM aggregation job"
```

---

### Task 8: Update run_inference.py for multi-model support

**Files:**
- Modify: `scripts/run_inference.py`
- Test: `tests/test_run_inference.py` (new file)

**Step 1: Write failing test**

Create `tests/test_run_inference.py`:

```python
"""Tests for multi-model inference script."""

import pytest


def test_get_model_class_returns_all_three():
    """get_model_class resolves all three model names."""
    from scripts.run_inference import get_model_class
    from blue_frogs.models.model_a import EfficientNetClassifier
    from blue_frogs.models.model_b_classifier import FusionClassifier
    from blue_frogs.models.model_c import FoundationModelClassifier

    assert get_model_class("model_a") is EfficientNetClassifier
    assert get_model_class("model_b") is FusionClassifier
    assert get_model_class("model_c") is FoundationModelClassifier


def test_get_model_class_raises_for_unknown():
    """get_model_class raises ValueError for unknown model."""
    from scripts.run_inference import get_model_class

    with pytest.raises(ValueError, match="Unknown model"):
        get_model_class("model_z")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_inference.py -v`
Expected: FAIL — `ImportError` for `get_model_class`

**Step 3: Update run_inference.py**

Replace the full content of `scripts/run_inference.py`:

```python
"""CLI script to run batch inference on the full iNat frog corpus."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from blue_frogs.config import RESULTS_DIR
from blue_frogs.inference.batch_scorer import run_batch_inference, filter_flagged_predictions
from blue_frogs.models.model_a import EfficientNetClassifier
from blue_frogs.models.model_b_classifier import FusionClassifier
from blue_frogs.models.model_c import FoundationModelClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "model_a": EfficientNetClassifier,
    "model_b": FusionClassifier,
    "model_c": FoundationModelClassifier,
}


def get_model_class(model_name: str):
    """Resolve model name to class."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_name]


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on frog images")
    parser.add_argument("--model", type=str, default="model_a",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model architecture to use")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Directory containing images to score")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="JSON file with image metadata")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-version", type=str, default=None)
    args = parser.parse_args()

    if args.model_version is None:
        args.model_version = f"{args.model}_v1"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_class = get_model_class(args.model)
    logger.info(f"Loading {args.model} from {args.checkpoint}")
    model = model_class.load_from_checkpoint(str(args.checkpoint))

    # Load metadata
    metadata = pd.read_json(args.metadata).to_dict("records")
    logger.info(f"Scoring {len(metadata)} images")

    # Run inference
    predictions = run_batch_inference(
        model=model,
        metadata=metadata,
        image_dir=args.image_dir,
        threshold=args.threshold,
        model_version=args.model_version,
        batch_size=args.batch_size,
    )

    # Save all predictions
    all_path = args.output_dir / f"predictions_{args.model_version}.csv"
    predictions.to_csv(all_path, index=False)
    logger.info(f"All predictions saved to {all_path}")

    # Save flagged predictions
    flagged = filter_flagged_predictions(predictions, args.threshold)
    flagged_path = args.output_dir / f"flagged_{args.model_version}.csv"
    flagged.to_csv(flagged_path, index=False)
    logger.info(f"Flagged {len(flagged)} observations above threshold {args.threshold}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_inference.py -v`
Expected: 2 PASSED

**Step 5: Run full test suite**

Run: `pytest`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add scripts/run_inference.py tests/test_run_inference.py
git commit -m "Update run_inference.py to support all three model architectures"
```

---

### Task 9: Add test predictions saving to train_fold

**Files:**
- Modify: `scripts/train.py` (add test prediction saving after each fold)

The aggregation script in Task 7 expects `fold_N/test_predictions.json` files. `train_fold` needs to save these after training completes.

**Step 1: Write failing test**

Add to `tests/test_train.py`:

```python
def test_save_test_predictions_creates_file(tmp_path):
    """save_test_predictions writes y_true and y_score to JSON."""
    from scripts.train import save_test_predictions

    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]
    save_test_predictions(y_true, y_score, tmp_path)

    pred_file = tmp_path / "test_predictions.json"
    assert pred_file.exists()
    data = json.loads(pred_file.read_text())
    assert data["y_true"] == y_true
    assert data["y_score"] == y_score
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_save_test_predictions_creates_file -v`
Expected: FAIL

**Step 3: Implement save_test_predictions**

Add to `scripts/train.py`, after the imports:

```python
def save_test_predictions(
    y_true: list, y_score: list, fold_dir: Path
) -> None:
    """Save test-set predictions for later aggregation."""
    import json
    pred_path = fold_dir / "test_predictions.json"
    with open(pred_path, "w") as f:
        json.dump({"y_true": y_true, "y_score": y_score}, f)
```

Then in `main()`, after training each fold and before saving to `results`, add test-set evaluation. In the fold loop, after `best_path = train_fold(...)` or `train_fold_two_stage(...)`:

```python
        # Evaluate on held-out test set and save predictions
        test_dataset = FrogDataset(test_meta, image_dir, transform=get_val_transforms())
        test_loader = DataLoader(
            test_dataset, batch_size=config["training"]["batch_size"],
            shuffle=False, num_workers=config["training"].get("num_workers", 4),
        )

        import torch
        loaded_model = model_class.load_from_checkpoint(best_path)
        loaded_model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                logits = loaded_model(images)
                probs = torch.sigmoid(logits.squeeze(-1))
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        fold_dir = output_dir / f"fold_{fold_idx}"
        save_test_predictions(all_labels, all_probs, fold_dir)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add scripts/train.py tests/test_train.py
git commit -m "Save test-set predictions per fold for aggregation"
```

---

### Task 10: Final integration verification

**Files:**
- No new files — verification only

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS (original 38 + new tests from Tasks 1-9)

**Step 2: Run linter on changed files**

Run: `ruff check scripts/train.py scripts/aggregate_folds.py scripts/run_inference.py tests/test_train.py tests/test_aggregate.py tests/test_run_inference.py`
Expected: No errors

**Step 3: Verify all SLURM scripts are well-formed**

Run: `bash -n scripts/slurm/submit_all.sh && bash -n scripts/slurm/train_model_a.sbatch && bash -n scripts/slurm/train_model_b.sbatch && bash -n scripts/slurm/train_model_c.sbatch && bash -n scripts/slurm/aggregate_results.sbatch && echo "All scripts valid"`
Expected: "All scripts valid"

**Step 4: Verify Dockerfile.gpu syntax**

Run: `docker build -f Dockerfile.gpu -t blue-frogs-gpu . --target builder 2>&1 | tail -1` (or just check it builds the first stage)

**Step 5: Commit any linter fixes**

If ruff found issues:
```bash
ruff check --fix scripts/ tests/
git add -u
git commit -m "Fix linter issues"
```

**Step 6: Use finishing-a-development-branch skill**

Invoke superpowers:finishing-a-development-branch to verify tests, present branch options.
