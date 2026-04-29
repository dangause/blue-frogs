# Model D: DINOv3 ViT-L/16 Classifier

**Date**: 2026-03-18
**Status**: Approved
**Goal**: Add a DINOv3-based classifier (Model D) to compare against the existing DINOv2 Model C baseline, targeting improved AUPRC on axanthism detection.

## Background

Model C (DINOv2 ViT-B/14, 86M params) is the current best performer at AUPRC 0.994. DINOv3 was released by Meta in February 2026 with significantly improved self-supervised features trained on 12x more data (1.7B images). The ViT-L/16 variant (300M params) is the target, given adequate GPU memory on the GB10 (128GB unified).

Note: The parameter increase (86M -> 300M) comes from moving up a model size tier (Base -> Large), not from the DINOv2-to-v3 upgrade itself. DINOv3 ViT-B/16 is also 86M params.

## Architecture

### Backbone
- **Model**: DINOv3 ViT-L/16 (300M parameters)
- **Loading**: HuggingFace Transformers (primary):
  ```python
  from transformers import AutoModel
  model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
  ```
  DINOv3's torch.hub API requires local checkpoints (`source='local'`, explicit `weights=` path), unlike DINOv2 which auto-downloads. HuggingFace is the simpler path.
- **Patch size**: 16 pixels (DINOv2 used 14)
- **Embedding dim**: Retrieved dynamically via `model.config.hidden_size`
- **CLS token**: Use the CLS token output for classification (same as DINOv2)

### Classification Head
Same proven architecture as Model C:
```
LayerNorm(embed_dim) -> Linear(embed_dim, 256) -> GELU -> Dropout(0.3) -> Linear(256, 1)
```

### Input Processing
- Image size: 384x384 pixels
- Padding: Defensive padding to nearest multiple of 16 (384 is already a multiple of 16, so no padding is applied at this image size; the logic is retained for robustness if image_size changes)
- Normalization: ImageNet mean/std
- Augmentation: Same pipeline as Model C (hue jitter limited to +/-5 degrees to preserve color signal)

### Design Decision: Separate Class vs. Extending Model C

`FoundationModelClassifier` in model_c.py already dispatches between dinov2 and bioclip backbones. Adding dinov3 there would be the minimal-diff approach.

We chose a separate `model_d.py` because:
1. **Clean A/B comparison**: Separate files ensure Model C (DINOv2) remains untouched as the baseline
2. **Different loading API**: DINOv3 uses HuggingFace Transformers vs DINOv2's torch.hub — this is a meaningful difference in dependencies and initialization
3. **Independent evolution**: Model D hyperparameters (batch size, model size tier) diverge from Model C

The trade-off is some code duplication (head construction, padding, configure_optimizers). This is acceptable for ~50 lines of shared logic.

## Training Strategy

### Two-Stage Training (same as Model C)

**Stage 1 - Linear Probe** (frozen backbone):
- Max epochs: 20
- Learning rate: 0.001
- Backbone: frozen (requires_grad=False)
- Purpose: Train classification head on DINOv3 features

**Stage 2 - Fine-tune** (unfrozen backbone):
- Max epochs: 80
- Differential learning rates:
  - Backbone: lr x 0.01 = 1e-5
  - Head: lr = 1e-3
- Scheduler: CosineAnnealingLR
- Early stopping: patience=10, monitor val/AUPRC

### Training Hyperparameters
- Loss: Focal (gamma=2.0, explicitly set in config)
- Optimizer: AdamW (weight_decay=0.01)
- Batch size: 16 (GB10 128GB unified memory)
- Precision: bf16-mixed (Blackwell BF16 support)
- Initial scope: Fold 0 only, expand to all 5 folds if results are promising

## Files to Create

### `src/blue_frogs/models/model_d.py` (New)
DINOv3 classifier class:
- `DINOv3Classifier(BaseClassifier)` with `_load_dinov3()` method
- Loads via `transformers.AutoModel.from_pretrained()`
- Patch size constant: `_DINOV3_PATCH_SIZE = 16`
- Support for model sizes via HuggingFace model IDs:
  - `"small"` -> `"facebook/dinov3-vits16-pretrain-lvd1689m"`
  - `"base"` -> `"facebook/dinov3-vitb16-pretrain-lvd1689m"`
  - `"large"` -> `"facebook/dinov3-vitl16-pretrain-lvd1689m"`
- Same `_pad_to_patch_multiple()` static method
- Same `configure_optimizers()` differential LR logic
- `forward()`: pad input, extract CLS token from model output, pass through head

### `configs/model_d.yaml` (New)
```yaml
model:
  backbone: dinov3
  model_size: large
  hidden_dim: 256
  dropout: 0.3
  pretrained: true
  focal_gamma: 2.0

training:
  loss_type: focal
  pos_weight: 1.0
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  max_epochs: 80
  batch_size: 16
  early_stopping_patience: 10
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

linear_probe:
  freeze_backbone: true
  max_epochs: 20
  learning_rate: 0.001

finetune:
  freeze_backbone: false
  backbone_lr_factor: 0.01
  max_epochs: 80

data:
  image_size: 384
  n_folds: 5
  seed: 42
```

## Files to Modify

### `scripts/train.py`
1. Add import: `from blue_frogs.models.model_d import DINOv3Classifier`
2. Register in MODEL_CLASSES: `"model_d": DINOv3Classifier`
3. Add `model_d` case to `build_model_kwargs()`:
   ```python
   elif model_name == "model_d":
       model_cfg = config["model"]
       return {
           **common_kwargs,
           "backbone": model_cfg["backbone"],
           "pretrained": model_cfg.get("pretrained", True),
           "model_size": model_cfg["model_size"],
           "hidden_dim": model_cfg["hidden_dim"],
           "dropout": model_cfg["dropout"],
           "freeze_backbone": model_cfg.get("freeze_backbone", False),
       }
   ```
4. Update `train_fold_two_stage` docstring from "Model C" to "foundation models"
5. Add `"model_d"` to `--model` choices in argparse

### `src/blue_frogs/models/__init__.py` (if it exports model classes)
Add `DINOv3Classifier` export if the existing `__init__.py` exports other model classes.

## Dependencies

DINOv3 loading requires `transformers` library (HuggingFace). Check if already in `pyproject.toml`; add if missing:
```
transformers >= 4.56.0
```

## Training Command

```bash
python scripts/train.py \
  --model model_d \
  --config configs/model_d.yaml \
  --data-dir data/ \
  --labels-file data/labels.json \
  --splits-file data/splits.json \
  --fold 0 \
  --precision bf16-mixed \
  --wandb-offline
```

## Success Criteria

- Model D trains without errors on fold 0
- AUPRC >= 0.994 (matching or exceeding Model C baseline)
- Training completes in reasonable time on GB10

## Risks

- **HuggingFace model availability**: The pretrained model ID `facebook/dinov3-vitl16-pretrain-lvd1689m` must be accessible from Spark. If behind a firewall, download weights first with `huggingface-cli download` and use a local path.
- **Memory**: ViT-L at batch_size=16 should fit in 128GB unified memory. Note: `nvidia-smi` may not report memory usage correctly on GB10's unified memory architecture; use `torch.cuda.memory_allocated()` for monitoring instead.
- **Embedding dimension**: DINOv3 ViT-L embed_dim may differ from DINOv2; retrieved dynamically via `model.config.hidden_size` to handle this.
- **CLS token extraction**: HuggingFace model output format differs from torch.hub; need to extract `outputs.last_hidden_state[:, 0]` for the CLS token.
