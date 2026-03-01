#!/bin/bash
#SBATCH --job-name=bf-model-c
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/model_c_fold%a_%j.log
#SBATCH --error=logs/model_c_fold%a_%j.err
#SBATCH --nodelist=alice

# Model C: DINOv2 two-stage (linear probe + fine-tune), 5-fold cross-validation
# Uses 48G memory — DINOv2 ViT-B/14 fine-tuning with full backbone unfrozen.

set -euo pipefail

FOLD=$SLURM_ARRAY_TASK_ID
SIF="${SIF:-$HOME/containers/blue-frogs-gpu.sif}"
DATA="${DATA_DIR:-$HOME/blue-frogs/data}"
OUT="${OUTPUT_DIR:-$HOME/blue-frogs/results}"
CONFIG="${CONFIG:-configs/model_c.yaml}"

echo "=== Model C | Fold $FOLD | $(date) ==="

apptainer exec --nv \
    --bind "$DATA":/app/data,"$OUT":/app/results \
    "$SIF" python scripts/train.py \
        --model model_c \
        --config "$CONFIG" \
        --data-dir /app/data \
        --labels-file /app/data/labels.json \
        --splits-file /app/data/splits.json \
        --output-dir /app/results \
        --fold "$FOLD" \
        --precision 16-mixed \
        --wandb-offline

echo "=== Model C | Fold $FOLD complete | $(date) ==="
