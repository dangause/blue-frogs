# Data Preparation Script & HPC Runbook Design

**Date:** 2026-03-02
**Status:** Approved
**Scope:** Data preparation script + HPC deployment runbook

## Context

HPC training infrastructure is complete (PR #4) but two things are needed before running on the cluster:
1. A script to generate `labels.json` and `splits.json` from existing downloaded data
2. A step-by-step runbook for deploying and running on the SLURM cluster

### Current Data State

- 2,365 observation folders in `data/raw_images/` (365 positives, 2,000 negatives)
- 3,260 total photos across observations
- Womack CSV has 369 iNat observation IDs; 365 matched on disk, 4 missing
- 1 observation folder is empty (0 photos)
- No `labels.json` or `splits.json` exists yet

## Component 1: `scripts/prepare_training_data.py`

Scans `data/raw_images/`, cross-references Womack CSV for positive labels, writes:

- **`data/labels.json`** — flat list of `{observation_id, photo_id, photo_path, label}` for all photos. Positives matched from Womack CSV, rest are label=0.
- **`data/splits.json`** — `{train_indices, test_indices}` from `create_stratified_splits()`. Split by observation to prevent leakage, then expanded to photo-level indices.

Uses existing `labels.py` (parse observation IDs) and `splits.py` (stratified splits). Skips empty observation dirs. Logs warnings for missing Womack positives.

CLI: `python scripts/prepare_training_data.py --data-dir data/ --labels-csv data/womack_labels.csv`

## Component 2: `docs/hpc-runbook.md`

Step-by-step deployment guide:
1. Run `prepare_training_data.py` locally
2. Build and push `Dockerfile.gpu`
3. Transfer data to cluster
4. Pull SIF on cluster
5. Run `submit_all.sh`
6. Monitor jobs
7. Retrieve results
