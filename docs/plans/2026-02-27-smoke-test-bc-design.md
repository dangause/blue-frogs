# Smoke Test Design: Models B and C

## Goal

Validate the full Model B (Fusion) and Model C (DINOv2) pipelines end-to-end with real data in Docker containers, reusing the same downloaded dataset from the Model A smoke test.

## Architecture

### Shared Helper Module

Extract the common pipeline steps from `scripts/smoke_test.py` into `scripts/smoke_test_common.py`:

- `step_download_labels(data_dir)` → DataFrame
- `step_fetch_positive_metadata(obs_ids)` → list[dict]
- `step_fetch_negative_metadata(positive_ids, n_negatives, max_pages)` → list[dict]
- `step_download_images(metadata, image_dir, max_workers)` → stats
- `step_build_metadata(positive_meta, negative_meta, image_dir)` → list[dict]
- `step_split(metadata, seed)` → (train_meta, val_meta)

Also extract the summary printing and metrics saving logic.

### Three Thin Scripts

- `scripts/smoke_test.py` — Model A (refactored to import from `smoke_test_common`)
- `scripts/smoke_test_b.py` — Model B (YOLO crop + LAB features + FusionClassifier)
- `scripts/smoke_test_c.py` — Model C (two-stage: linear probe → fine-tune)

### Data Sharing

All three models reuse the same `data/raw_images/` and `data/womack_labels.csv`. Model B additionally writes YOLO crops to `data/cropped_images/{obs_id}/{photo_id}.jpg`. The same train/val split (seed=42, test_fraction=0.2) is used across all models for comparable evaluation.

---

## Model B Pipeline

### Step 1: YOLO Crop Preprocessing

After the shared download step, run `FrogDetector.crop_frog()` on every raw image. Save crops to `data/cropped_images/`. Skip images that already have a cached crop.

Config parameters from `configs/smoke_test_b.yaml`:
- `detector.conf_threshold: 0.5`
- `detector.padding_fraction: 0.1`

### Step 2: Dataset with Color Features

Create a `FrogDatasetWithColorFeatures` class (in the smoke test script, not the library) that:
1. Reads the cropped image (falls back to raw if no crop exists)
2. Computes `extract_lab_features(image_rgb)` on the raw uint8 image before transforms
3. Applies albumentations transforms to the image
4. Returns `(image_tensor, color_feat_tensor, label)`

### Step 3: Train FusionClassifier

Instantiate `FusionClassifier` with config parameters. Use the 3-tuple DataLoader. Train for 3 epochs.

### Step 4: Evaluate

3-tuple-aware eval loop: `model(images, color_feats)` → sigmoid → metrics.

### Dependencies

Dockerfile needs `ultralytics` added for YOLO. The `yolov8m.pt` weights auto-download on first run.

---

## Model C Pipeline

### Step 1: Standard Dataset

Reuse `FrogDataset` with `get_train_transforms()` / `get_val_transforms()` — identical to Model A.

### Step 2: Stage 1 — Linear Probe

Instantiate `FoundationModelClassifier(freeze_backbone=True, backbone="dinov2", model_size="base")`. Train for 2 epochs. Save best checkpoint.

### Step 3: Stage 2 — Fine-Tune

Load the best checkpoint from Stage 1. Create a new `FoundationModelClassifier(freeze_backbone=False)` and load state dict. Train for 2 more epochs with differential LR (backbone gets lr * 0.01). Save best checkpoint.

### Step 4: Evaluate

Standard eval loop (same as Model A) on the Stage 2 best checkpoint.

### Dependencies

DINOv2 weights auto-download via `torch.hub` on first run. No extra pip packages needed.

---

## Configs

### `configs/smoke_test_b.yaml`

```yaml
detector:
  model: yolov8m.pt
  conf_threshold: 0.5
  padding_fraction: 0.1

classifier:
  backbone: tf_efficientnetv2_s
  pretrained: true
  color_feature_dim: 30
  fusion_hidden: 256
  dropout: 0.3

training:
  loss_type: weighted_bce
  pos_weight: 20.0
  optimizer: sgd
  learning_rate: 0.001
  weight_decay: 0.0
  max_epochs: 3
  batch_size: 8
  num_workers: 0
  early_stopping_patience: 3
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

data:
  image_size: 384
  seed: 42
  n_negatives: 2000

smoke_test:
  max_negative_pages: 50
```

### `configs/smoke_test_c.yaml`

```yaml
model:
  backbone: dinov2
  model_size: base
  hidden_dim: 256
  dropout: 0.3

linear_probe:
  freeze_backbone: true
  max_epochs: 2
  learning_rate: 0.001

finetune:
  freeze_backbone: false
  max_epochs: 2
  backbone_lr_factor: 0.01

training:
  loss_type: weighted_bce
  pos_weight: 20.0
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  batch_size: 8
  num_workers: 0
  early_stopping_patience: 3
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

data:
  image_size: 384
  seed: 42
  n_negatives: 2000

smoke_test:
  max_negative_pages: 50
```

## Docker

The existing Dockerfile entrypoint stays as Model A. Models B and C override the entrypoint:

```bash
# Model B
docker run --shm-size=2g -m 8g -v $(pwd)/data:/app/data \
  blue-frogs-smoke python scripts/smoke_test_b.py --data-dir /app/data --output-dir /app/data/smoke_output_b

# Model C
docker run --shm-size=2g -m 8g -v $(pwd)/data:/app/data \
  blue-frogs-smoke python scripts/smoke_test_c.py --data-dir /app/data --output-dir /app/data/smoke_output_c
```

The Dockerfile pip install step needs `ultralytics` added for Model B.
