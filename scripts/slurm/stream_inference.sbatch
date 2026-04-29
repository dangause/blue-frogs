#!/bin/bash
#SBATCH --job-name=bf-stream
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stream_inference_%j.log
#SBATCH --error=logs/stream_inference_%j.err
#SBATCH --nodelist=alice

# Streaming batch inference: fetch → score → delete in bounded batches.
# Resumable — resubmit the same script after wall-time kill and it picks
# up from the last completed batch via the state file.

set -euo pipefail

MODEL="${MODEL:-model_a}"
CHECKPOINT="${CHECKPOINT:-models/model_a_fold0_best.ckpt}"
MODEL_VERSION="${MODEL_VERSION:-${MODEL}_v1}"
THRESHOLD="${THRESHOLD:-0.5}"
OBS_PER_BATCH="${OBS_PER_BATCH:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_WORKERS="${MAX_WORKERS:-8}"

SIF="${SIF:-$HOME/containers/blue-frogs-gpu.sif}"
RESULTS="${OUTPUT_DIR:-$HOME/blue-frogs/results/streaming}"
TMP_DIR="${TMP_DIR:-/scratch/$USER/blue_frogs_stream}"
DETECTOR_CKPT="${DETECTOR_CKPT:-}"

echo "=== Streaming Inference | ${MODEL} | $(date) ==="
echo "SIF:        $SIF"
echo "Checkpoint: $CHECKPOINT"
echo "Results:    $RESULTS"
echo "Tmp dir:    $TMP_DIR"

mkdir -p "$RESULTS" "$TMP_DIR" logs

# Build detector flag for Model B
DETECTOR_FLAG=""
if [ "$MODEL" = "model_b" ] && [ -n "$DETECTOR_CKPT" ]; then
    DETECTOR_FLAG="--detector-checkpoint /app/results/${DETECTOR_CKPT}"
fi

apptainer exec --nv \
    --bind "$RESULTS":/app/results,"$TMP_DIR":/tmp/blue_frogs_stream \
    "$SIF" python scripts/stream_inference.py \
        --model "$MODEL" \
        --checkpoint "/app/results/$CHECKPOINT" \
        --threshold "$THRESHOLD" \
        --obs-per-batch "$OBS_PER_BATCH" \
        --inference-batch-size "$BATCH_SIZE" \
        --output-dir /app/results \
        --state-file /app/results/state.json \
        --tmp-dir /tmp/blue_frogs_stream \
        --max-workers "$MAX_WORKERS" \
        --model-version "$MODEL_VERSION" \
        $DETECTOR_FLAG

echo "=== Streaming Inference complete | $(date) ==="
