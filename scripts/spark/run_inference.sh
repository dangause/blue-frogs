#!/bin/bash
# Streaming inference for direct GPU execution (no Docker).
#
# Usage:
#   bash scripts/spark/run_inference.sh
#
# Override defaults:
#   MODEL=model_a bash scripts/spark/run_inference.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results}"
MODEL="${MODEL:-model_b}"
CHECKPOINT="${CHECKPOINT:-}"
MODEL_VERSION="${MODEL_VERSION:-${MODEL}_v1}"
THRESHOLD="${THRESHOLD:-0.5}"
OBS_PER_BATCH="${OBS_PER_BATCH:-1000}"
MAX_BATCHES="${MAX_BATCHES:-}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "${PROJECT_DIR}/.venv" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

# Auto-detect best checkpoint if not specified
if [ -z "$CHECKPOINT" ]; then
    if [ "$MODEL" = "model_b" ]; then
        CHECKPOINT=$(find "$RESULTS_DIR/$MODEL" -path "*/finetune/best-*.ckpt" | head -1 || true)
    fi
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=$(find "$RESULTS_DIR/$MODEL" -name "best-*.ckpt" | head -1 || true)
    fi
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found for $MODEL in $RESULTS_DIR/$MODEL"
        exit 1
    fi
fi

echo "================================================"
echo "Streaming Inference: $MODEL"
echo "================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Threshold:  $THRESHOLD"
echo "================================================"

CMD="python scripts/stream_inference.py \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --threshold $THRESHOLD \
    --obs-per-batch $OBS_PER_BATCH \
    --output-dir $RESULTS_DIR/streaming_${MODEL_VERSION} \
    --model-version $MODEL_VERSION"

if [ -n "${MAX_BATCHES:-}" ]; then
    CMD="$CMD --max-batches $MAX_BATCHES"
fi

eval $CMD 2>&1 | tee "${LOG_DIR}/inference_${MODEL_VERSION}.log"

echo ""
echo "Inference complete. Results in: $RESULTS_DIR/streaming_${MODEL_VERSION}/"
