#!/bin/bash
# Streaming inference over full iNat frog corpus using Docker.
# Run AFTER training is complete.
#
# Usage:
#   bash scripts/docker/run_inference.sh
#
# Override defaults:
#   MODEL=model_a CHECKPOINT=results/model_a/fold_0/best.ckpt bash scripts/docker/run_inference.sh

set -euo pipefail

# --- Configuration ---
IMAGE="${IMAGE:-blue-frogs-gpu:latest}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results}"
GPU_ID="${GPU_ID:-0}"
MODEL="${MODEL:-model_b}"
CHECKPOINT="${CHECKPOINT:-}"
DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-}"
MODEL_VERSION="${MODEL_VERSION:-${MODEL}_v1}"
THRESHOLD="${THRESHOLD:-0.5}"
OBS_PER_BATCH="${OBS_PER_BATCH:-1000}"
MAX_BATCHES="${MAX_BATCHES:-}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Auto-detect best checkpoint if not specified
if [ -z "$CHECKPOINT" ]; then
    if [ "$MODEL" = "model_b" ]; then
        # Model B uses two-stage, check finetune first
        CHECKPOINT=$(find "$RESULTS_DIR/$MODEL" -path "*/finetune/best-*.ckpt" | head -1 || true)
    fi
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=$(find "$RESULTS_DIR/$MODEL" -name "best-*.ckpt" | head -1 || true)
    fi
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found for $MODEL in $RESULTS_DIR/$MODEL"
        echo "  Specify manually: CHECKPOINT=/path/to/best.ckpt bash $0"
        exit 1
    fi
    # Convert to container path
    CHECKPOINT="/app/results/${CHECKPOINT#"$RESULTS_DIR/"}"
fi

echo "================================================"
echo "Streaming Inference: $MODEL"
echo "================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Threshold:  $THRESHOLD"
echo "Batch size: $OBS_PER_BATCH obs/batch"
echo "================================================"

# Build inference command
INFERENCE_CMD="python scripts/stream_inference.py \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --threshold $THRESHOLD \
    --obs-per-batch $OBS_PER_BATCH \
    --output-dir /app/results/streaming_${MODEL_VERSION} \
    --model-version $MODEL_VERSION"

# Add detector for Model B
if [ "$MODEL" = "model_b" ]; then
    if [ -n "$DETECTOR_CHECKPOINT" ]; then
        INFERENCE_CMD="$INFERENCE_CMD --detector-checkpoint $DETECTOR_CHECKPOINT"
    fi
fi

# Add max batches if specified
if [ -n "$MAX_BATCHES" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --max-batches $MAX_BATCHES"
fi

docker run --rm --gpus "device=${GPU_ID}" \
    -v "${DATA_DIR}":/app/data \
    -v "${RESULTS_DIR}":/app/results \
    "${IMAGE}" \
    $INFERENCE_CMD \
    2>&1 | tee "${LOG_DIR}/inference_${MODEL_VERSION}.log"

echo ""
echo "Inference complete. Results in: $RESULTS_DIR/streaming_${MODEL_VERSION}/"
