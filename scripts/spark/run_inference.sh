#!/bin/bash
# Streaming inference for direct GPU execution (no Docker).
#
# Usage:
#   bash scripts/spark/run_inference.sh
#
# Override defaults:
#   MODELS="model_c model_d" bash scripts/spark/run_inference.sh
#   MODELS=model_a SAVE_FLAGGED=1 bash scripts/spark/run_inference.sh
#   CALIBRATION_FILE=results/calibration.json bash scripts/spark/run_inference.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results}"
MODELS="${MODELS:-model_c}"
CHECKPOINT="${CHECKPOINT:-}"
CALIBRATION_FILE="${CALIBRATION_FILE:-}"
MODEL_VERSION="${MODEL_VERSION:-${MODELS// /_}_v1}"
THRESHOLD="${THRESHOLD:-0.5}"
OBS_PER_BATCH="${OBS_PER_BATCH:-1000}"
MAX_BATCHES="${MAX_BATCHES:-}"
SAVE_FLAGGED="${SAVE_FLAGGED:-}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "${PROJECT_DIR}/.venv" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

# Auto-detect checkpoints if not specified
CHECKPOINTS=""
if [ -z "$CHECKPOINT" ]; then
    for MODEL in $MODELS; do
        CKPT=""
        if [ "$MODEL" = "model_b" ]; then
            CKPT=$(find "$RESULTS_DIR/$MODEL" "$PROJECT_DIR/models/$MODEL" -path "*/finetune/best-*.ckpt" 2>/dev/null | head -1 || true)
        fi
        if [ -z "$CKPT" ]; then
            CKPT=$(find "$RESULTS_DIR/$MODEL" "$PROJECT_DIR/models/$MODEL" -name "best-*.ckpt" 2>/dev/null | head -1 || true)
        fi
        if [ -z "$CKPT" ]; then
            echo "ERROR: No checkpoint found for $MODEL in $RESULTS_DIR/$MODEL or models/$MODEL"
            exit 1
        fi
        CHECKPOINTS="$CHECKPOINTS $CKPT"
    done
else
    CHECKPOINTS="$CHECKPOINT"
fi

echo "================================================"
echo "Streaming Inference"
echo "================================================"
echo "Models:     $MODELS"
echo "Checkpoints:$CHECKPOINTS"
echo "Threshold:  $THRESHOLD"
if [ -n "$CALIBRATION_FILE" ]; then
    echo "Calibration: $CALIBRATION_FILE"
fi
echo "================================================"

CMD="python scripts/stream_inference.py \
    --models $MODELS \
    --checkpoints $CHECKPOINTS \
    --threshold $THRESHOLD \
    --obs-per-batch $OBS_PER_BATCH \
    --output-dir $RESULTS_DIR/streaming_${MODEL_VERSION}"

if [ -n "${MAX_BATCHES:-}" ]; then
    CMD="$CMD --max-batches $MAX_BATCHES"
fi

if [ -n "${CALIBRATION_FILE:-}" ]; then
    CMD="$CMD --calibration-file $CALIBRATION_FILE"
fi

if [ -n "${SAVE_FLAGGED:-}" ]; then
    CMD="$CMD --save-flagged"
fi

eval $CMD 2>&1 | tee "${LOG_DIR}/inference_${MODEL_VERSION}.log"

echo ""
echo "Inference complete. Results in: $RESULTS_DIR/streaming_${MODEL_VERSION}/"
