# HPC Deployment Runbook

Step-by-step guide for training the axanthism classifier models on the CAS SLURM cluster.

## Prerequisites

- Docker installed locally
- Docker Hub account (for pushing GPU image)
- SSH access to the SLURM cluster (`alice` node)
- Apptainer available on the cluster
- Downloaded images in `data/raw_images/` (2,365 observations)
- Womack labels CSV at `data/womack_labels.csv`

## Step 1: Prepare Training Data (Local)

Generate `labels.json` and `splits.json` from the downloaded images:

```bash
python scripts/prepare_training_data.py \
    --data-dir data/ \
    --labels-csv data/womack_labels.csv
```

Verify output:

```bash
python -c "import json; d=json.load(open('data/labels.json')); print(f'{len(d)} photos, {sum(1 for e in d if e[\"label\"]==1)} positive')"
python -c "import json; d=json.load(open('data/splits.json')); print(f'train: {len(d[\"train_indices\"])}, test: {len(d[\"test_indices\"])}')"
```

Expected: ~365 positive photos, ~2000 negative photos, 80/20 train/test split.

## Step 2: Build and Push GPU Docker Image

```bash
docker build -f Dockerfile.gpu -t dangause/blue-frogs-gpu:latest .
docker push dangause/blue-frogs-gpu:latest
```

## Step 3: Transfer Data to Cluster

```bash
# Create directory structure on cluster
ssh alice "mkdir -p ~/blue-frogs/data ~/blue-frogs/results ~/containers"

# Transfer training data and metadata
rsync -avz --progress data/raw_images/ alice:~/blue-frogs/data/raw_images/
scp data/labels.json data/splits.json alice:~/blue-frogs/data/
```

## Step 4: Pull Container on Cluster

```bash
ssh alice "apptainer pull ~/containers/blue-frogs-gpu.sif docker://dangause/blue-frogs-gpu:latest"
```

Verify the image works:

```bash
ssh alice "apptainer exec --nv ~/containers/blue-frogs-gpu.sif python -c 'import torch; print(torch.cuda.is_available())'"
```

Expected: `True`

## Step 5: Submit Training Jobs

Copy SLURM scripts to cluster and submit:

```bash
rsync -avz scripts/slurm/ alice:~/blue-frogs/scripts/slurm/
rsync -avz configs/ alice:~/blue-frogs/configs/
```

Submit all jobs (3 model arrays + dependency-chained aggregation):

```bash
ssh alice "cd ~/blue-frogs && mkdir -p logs && bash scripts/slurm/submit_all.sh"
```

This submits:
- **Model A** (EfficientNetV2-S): 5 folds, 32G RAM each, GPU
- **Model B** (YOLO+CNN-LAB fusion): 5 folds, 32G RAM each, GPU
- **Model C** (DINOv2): 5 folds, 48G RAM each, GPU
- **Aggregation**: runs automatically after all training completes (CPU, 16G)

### Environment Variable Overrides

Override defaults by setting environment variables before `submit_all.sh`:

```bash
SIF=~/containers/blue-frogs-gpu.sif \
DATA_DIR=~/blue-frogs/data \
OUTPUT_DIR=~/blue-frogs/results \
bash scripts/slurm/submit_all.sh
```

## Step 6: Monitor Progress

```bash
# Check job status
ssh alice "squeue -u \$USER"

# Watch logs in real-time (replace JOBID and FOLD)
ssh alice "tail -f ~/blue-frogs/logs/model_a_fold0_JOBID.log"

# Check for errors
ssh alice "ls -la ~/blue-frogs/logs/*.err"
ssh alice "grep -l 'Error\|OOM\|CUDA' ~/blue-frogs/logs/*.err"
```

## Step 7: Retrieve Results

After the aggregation job completes:

```bash
# Get the comparison summary
scp alice:~/blue-frogs/results/comparison_summary.md results/

# Get all results (checkpoints, predictions, metrics)
rsync -avz alice:~/blue-frogs/results/ results/
```

The `comparison_summary.md` contains per-model AUPRC, AUROC, F1 with bootstrap CIs and McNemar's test results.

## Troubleshooting

### OOM (Out of Memory)

If a job fails with CUDA OOM errors:

- **Model C** already uses 48G; try reducing batch size in `configs/model_c.yaml`
- Reduce `batch_size` in the relevant config YAML
- Check with: `grep -i 'oom\|memory' logs/*.err`

### Missing SIF Container

```
FATAL:   While making image from oci registry: ...
```

Re-pull the container:

```bash
ssh alice "apptainer pull --force ~/containers/blue-frogs-gpu.sif docker://dangause/blue-frogs-gpu:latest"
```

### YOLO Weights Not Found (Model B)

Model B requires YOLOv8 weights. If they fail to auto-download on the cluster:

1. Download locally: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`
2. Transfer: `scp yolov8n.pt alice:~/blue-frogs/data/`
3. Update `configs/model_b.yaml` to point to the weights path

### Jobs Stuck in Pending

```bash
# Check why jobs are pending
ssh alice "squeue -u \$USER -o '%.10i %.9P %.8j %.8u %.2t %.10M %.6D %R'"
```

Common reasons: GPU not available, node down, resource limits reached.

### WandB Offline Sync

Training runs with `--wandb-offline`. To sync after completion:

```bash
ssh alice "cd ~/blue-frogs && apptainer exec ~/containers/blue-frogs-gpu.sif wandb sync results/wandb/offline-*"
```
