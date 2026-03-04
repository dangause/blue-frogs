#!/bin/bash
# Direct setup for ARM64 NVIDIA GPU nodes (Jetson/Grace) — no Docker needed.
#
# Usage:
#   bash scripts/setup_spark.sh
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - NVIDIA GPU with CUDA drivers installed
#   - Data directory with: labels.json, splits.json, raw_images/

set -euo pipefail

echo "================================================"
echo "Blue Frogs — Spark Node Setup"
echo "================================================"
echo "Arch:   $(uname -m)"
echo "Python: $(python3 --version)"
echo "CUDA:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'not found')"
echo "================================================"

# Create virtualenv
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtualenv..."
    python3 -m venv .venv
else
    echo "[1/4] Virtualenv already exists"
fi

source .venv/bin/activate

# Install PyTorch (ARM64-compatible via pip)
echo "[2/4] Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project with all extras
echo "[3/4] Installing blue-frogs..."
pip install -e ".[bioclip,yolo,dev]"

# Pre-download YOLO weights
echo "[4/4] Downloading YOLOv8m weights..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

echo ""
echo "================================================"
echo "Setup complete! Activate with: source .venv/bin/activate"
echo ""
echo "Train all models:"
echo "  bash scripts/spark/train_all.sh"
echo ""
echo "Run inference:"
echo "  bash scripts/spark/run_inference.sh"
echo "================================================"
