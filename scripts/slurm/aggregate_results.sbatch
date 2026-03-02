#!/bin/bash
#SBATCH --job-name=bf-aggregate
#SBATCH --mem=16G
#SBATCH --output=logs/aggregate_%j.log
#SBATCH --error=logs/aggregate_%j.err

# Aggregation runs on CPU after all training jobs complete.
# Submitted with --dependency=afterok by submit_all.sh.

set -euo pipefail

SIF="${SIF:-$HOME/containers/blue-frogs-gpu.sif}"
OUT="${OUTPUT_DIR:-$HOME/blue-frogs/results}"

echo "=== Aggregation | $(date) ==="

apptainer exec \
    --bind "$OUT":/app/results \
    "$SIF" python scripts/aggregate_folds.py \
        --results-dir /app/results \
        --output-dir /app/results

echo "=== Aggregation complete | $(date) ==="
echo "Results at: $OUT/comparison_summary.md"
