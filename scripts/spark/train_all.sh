#!/bin/bash
# Full training pipeline for direct GPU execution (no Docker).
#
# Usage:
#   bash scripts/spark/train_all.sh
#
# Override defaults:
#   DATA_DIR=/data/frogs RESULTS_DIR=/results bash scripts/spark/train_all.sh

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results}"
PRECISION="${PRECISION:-16-mixed}"
N_FOLDS="${N_FOLDS:-5}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Activate virtualenv if not already active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "${PROJECT_DIR}/.venv" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

echo "================================================"
echo "Blue Frogs Training Pipeline"
echo "================================================"
echo "Data:    $DATA_DIR"
echo "Results: $RESULTS_DIR"
echo "Folds:   $N_FOLDS"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "================================================"

export WANDB_MODE=offline

# --- Step 0: Preprocess YOLO crops for Model B ---
echo ""
echo "[Step 0] Preprocessing YOLO crops for Model B..."
python scripts/preprocess_crops.py \
    --data-dir "$DATA_DIR" \
    --labels-file "$DATA_DIR/labels.json" \
    2>&1 | tee "${LOG_DIR}/preprocess_crops.log"

# --- Step 1: Train Model A (5 folds) ---
echo ""
echo "[Step 1] Training Model A (${N_FOLDS} folds)..."
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    echo "  --- Model A | Fold ${FOLD} | $(date) ---"
    python scripts/train.py \
        --model model_a \
        --config configs/model_a.yaml \
        --data-dir "$DATA_DIR" \
        --labels-file "$DATA_DIR/labels.json" \
        --splits-file "$DATA_DIR/splits.json" \
        --output-dir "$RESULTS_DIR" \
        --fold "$FOLD" \
        --precision "$PRECISION" \
        --wandb-offline \
        2>&1 | tee "${LOG_DIR}/model_a_fold${FOLD}.log"
    echo "  --- Model A | Fold ${FOLD} complete | $(date) ---"
done

# --- Step 2: Train Model B (5 folds) ---
echo ""
echo "[Step 2] Training Model B (${N_FOLDS} folds)..."
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    echo "  --- Model B | Fold ${FOLD} | $(date) ---"
    python scripts/train.py \
        --model model_b \
        --config configs/model_b.yaml \
        --data-dir "$DATA_DIR" \
        --labels-file "$DATA_DIR/labels.json" \
        --splits-file "$DATA_DIR/splits.json" \
        --output-dir "$RESULTS_DIR" \
        --fold "$FOLD" \
        --precision "$PRECISION" \
        --crop-dir "$DATA_DIR/cropped_images" \
        --lab-features "$DATA_DIR/crop_lab_features.npy" \
        --lab-index "$DATA_DIR/crop_lab_index.json" \
        --wandb-offline \
        2>&1 | tee "${LOG_DIR}/model_b_fold${FOLD}.log"
    echo "  --- Model B | Fold ${FOLD} complete | $(date) ---"
done

# --- Step 3: Train Model C (5 folds) ---
echo ""
echo "[Step 3] Training Model C (${N_FOLDS} folds)..."
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    echo "  --- Model C | Fold ${FOLD} | $(date) ---"
    python scripts/train.py \
        --model model_c \
        --config configs/model_c.yaml \
        --data-dir "$DATA_DIR" \
        --labels-file "$DATA_DIR/labels.json" \
        --splits-file "$DATA_DIR/splits.json" \
        --output-dir "$RESULTS_DIR" \
        --fold "$FOLD" \
        --precision "$PRECISION" \
        --wandb-offline \
        2>&1 | tee "${LOG_DIR}/model_c_fold${FOLD}.log"
    echo "  --- Model C | Fold ${FOLD} complete | $(date) ---"
done

# --- Step 4: Aggregate results ---
echo ""
echo "[Step 4] Aggregating fold results..."
python scripts/aggregate_folds.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR" \
    2>&1 | tee "${LOG_DIR}/aggregate.log"

echo ""
echo "================================================"
echo "Training complete! Results in: $RESULTS_DIR"
echo "================================================"
echo ""
echo "Next: run inference with scripts/spark/run_inference.sh"
