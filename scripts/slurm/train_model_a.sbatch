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
