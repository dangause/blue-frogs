#!/bin/bash
# Submit all model training jobs with dependency-chained aggregation.
#
# Usage: bash scripts/slurm/submit_all.sh
#
# Override defaults with environment variables:
#   SIF=~/containers/blue-frogs-gpu.sif DATA_DIR=~/data OUTPUT_DIR=~/results bash scripts/slurm/submit_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

echo "Submitting training jobs..."

JOB_A=$(sbatch --parsable "$SCRIPT_DIR/train_model_a.sbatch")
echo "  Model A: job $JOB_A (array 0-4)"

JOB_B=$(sbatch --parsable "$SCRIPT_DIR/train_model_b.sbatch")
echo "  Model B: job $JOB_B (array 0-4)"

JOB_C=$(sbatch --parsable "$SCRIPT_DIR/train_model_c.sbatch")
echo "  Model C: job $JOB_C (array 0-4)"

echo ""
echo "Submitting aggregation job (depends on all training)..."
AGG=$(sbatch --parsable \
    --dependency="afterok:$JOB_A:$JOB_B:$JOB_C" \
    "$SCRIPT_DIR/aggregate_results.sbatch")
echo "  Aggregation: job $AGG"

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $JOB_A $JOB_B $JOB_C $AGG"
