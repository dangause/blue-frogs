#!/bin/bash
# Full training + inference pipeline for Docker GPU node (no SLURM).
#
# Usage:
#   bash scripts/docker/train_all.sh
#
# Override defaults with environment variables:
#   DATA_DIR=/data/frogs RESULTS_DIR=/results GPU_ID=0 bash scripts/docker/train_all.sh
#
# Prerequisites:
#   - Docker with NVIDIA runtime (nvidia-container-toolkit)
#   - Data directory with: labels.json, splits.json, raw_images/
#   - Built image: docker build -f Dockerfile.gpu -t blue-frogs-gpu .

set -euo pipefail

# --- Configuration ---
IMAGE="${IMAGE:-blue-frogs-gpu:latest}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results}"
GPU_ID="${GPU_ID:-0}"
PRECISION="${PRECISION:-16-mixed}"
N_FOLDS="${N_FOLDS:-5}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Common docker run flags
DOCKER_RUN="docker run --rm --gpus device=${GPU_ID} \
  -v ${DATA_DIR}:/app/data \
  -v ${RESULTS_DIR}:/app/results \
  -e WANDB_MODE=offline \
  ${IMAGE}"

echo "================================================"
echo "Blue Frogs Training Pipeline"
echo "================================================"
echo "Image:   $IMAGE"
echo "Data:    $DATA_DIR"
echo "Results: $RESULTS_DIR"
echo "GPU:     $GPU_ID"
echo "Folds:   $N_FOLDS"
echo "================================================"

# --- Step 0: Preprocess YOLO crops for Model B ---
echo ""
echo "[Step 0] Preprocessing YOLO crops for Model B..."
$DOCKER_RUN python scripts/preprocess_crops.py \
    --data-dir /app/data \
    --labels-file /app/data/labels.json \
    2>&1 | tee "${LOG_DIR}/preprocess_crops.log"

# --- Step 1: Train Model A (5 folds) ---
echo ""
echo "[Step 1] Training Model A (${N_FOLDS} folds)..."
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    echo "  --- Model A | Fold ${FOLD} | $(date) ---"
    $DOCKER_RUN python scripts/train.py \
        --model model_a \
        --config configs/model_a.yaml \
        --data-dir /app/data \
        --labels-file /app/data/labels.json \
        --splits-file /app/data/splits.json \
        --output-dir /app/results \
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
    $DOCKER_RUN python scripts/train.py \
        --model model_b \
        --config configs/model_b.yaml \
        --data-dir /app/data \
        --labels-file /app/data/labels.json \
        --splits-file /app/data/splits.json \
        --output-dir /app/results \
        --fold "$FOLD" \
        --precision "$PRECISION" \
        --crop-dir /app/data/cropped_images \
        --lab-features /app/data/crop_lab_features.npy \
        --lab-index /app/data/crop_lab_index.json \
        --wandb-offline \
        2>&1 | tee "${LOG_DIR}/model_b_fold${FOLD}.log"
    echo "  --- Model B | Fold ${FOLD} complete | $(date) ---"
done

# --- Step 3: Train Model C (5 folds) ---
echo ""
echo "[Step 3] Training Model C (${N_FOLDS} folds)..."
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    echo "  --- Model C | Fold ${FOLD} | $(date) ---"
    $DOCKER_RUN python scripts/train.py \
        --model model_c \
        --config configs/model_c.yaml \
        --data-dir /app/data \
        --labels-file /app/data/labels.json \
        --splits-file /app/data/splits.json \
        --output-dir /app/results \
        --fold "$FOLD" \
        --precision "$PRECISION" \
        --wandb-offline \
        2>&1 | tee "${LOG_DIR}/model_c_fold${FOLD}.log"
    echo "  --- Model C | Fold ${FOLD} complete | $(date) ---"
done

# --- Step 4: Aggregate results ---
echo ""
echo "[Step 4] Aggregating fold results..."
$DOCKER_RUN python scripts/aggregate_folds.py \
    --results-dir /app/results \
    --output-dir /app/results \
    2>&1 | tee "${LOG_DIR}/aggregate.log"

echo ""
echo "================================================"
echo "Training complete! Results in: $RESULTS_DIR"
echo "================================================"
echo ""
echo "Next: run inference with scripts/docker/run_inference.sh"
