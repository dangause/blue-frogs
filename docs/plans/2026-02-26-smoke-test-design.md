# Local Smoke Test Design

**Date:** 2026-02-26
**Status:** Approved

## Goal

Verify the full axanthism classifier pipeline works end-to-end with real data on a local machine before moving to HPC. Runs inside a Docker container for reproducibility.

## Hardware

- Apple M3 Max, 36GB RAM
- MPS available natively but not inside Docker (CPU fallback)

## Architecture

Single Docker container running the full pipeline, orchestrated by `scripts/smoke_test.sh`. Data cached on a host volume so re-runs skip downloads.

## Data

- All ~372 Womack positives
- ~2000 tiered negatives (hard/medium/easy)
- ~15 min download time (first run only)

## Components

### Dockerfile (repo root)

- Base: `python:3.11-slim`
- Multi-stage build: build stage installs deps, runtime stage is slim
- System deps: `libgl1-mesa-glx`, `libglib2.0-0` (OpenCV)
- Entrypoint: `scripts/smoke_test.sh`
- Volume: `/app/data` for persistent data caching

### scripts/smoke_test.sh

Sequential pipeline steps with early-exit-on-failure:

1. Download Womack labels (skip if cached)
2. Fetch observation metadata from iNat API for positives
3. Sample ~2000 negatives from iNat (tiered)
4. Download images (parallel, skip existing)
5. Create stratified train/val split (80/20, single fold)
6. Train Model A: 3 epochs, batch size 16, CPU
7. Evaluate on val set
8. Print metrics: AUROC, AUPRC, precision@90recall

### configs/smoke_test.yaml

- `max_epochs: 3`
- `batch_size: 16`
- `num_workers: 2`
- `accelerator: auto`
- `n_folds: 1`
- `image_size: 384`
- Model A only (EfficientNetV2-S, pretrained)
- W&B disabled

### Docker commands

```bash
docker build -t blue-frogs-smoke .
docker run -v $(pwd)/data:/app/data blue-frogs-smoke
```

## Success Criteria

- All pipeline stages complete without error
- Model trains for 3 epochs with decreasing loss
- Validation metrics computed and printed
- No crashes or OOM errors

## Scope Exclusions

- Models B and C
- Full k-fold cross-validation
- W&B logging
- Figure generation
- Batch inference on full corpus
