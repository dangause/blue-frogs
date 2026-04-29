# Model D Findings: DINOv3 ViT-L/16

## Motivation

Models A-C established strong baselines for axanthism detection, with Model C (DINOv2-Base, 86M params) achieving the best performance (AUPRC 0.994). Model D tests whether a larger, newer foundation model improves on these results.

DINOv3 was trained on LVD-1.6B (1.6 billion images, ~12x DINOv2's 142M) and the ViT-L/16 variant has 300M parameters (3.5x Model C's 86M). The hypothesis was that richer representations from more data and a larger model would improve discrimination in edge cases, particularly reducing false positives on naturally blue species.

## Architecture

| Component | Model C (DINOv2) | Model D (DINOv3) |
|-----------|-----------------|-----------------|
| Backbone | DINOv2 ViT-B/14 | DINOv3 ViT-L/16 |
| Parameters | 86M | 300M |
| Training data | LVD-142M | LVD-1.6B |
| Feature dim | 768 | 1024 |
| Patch size | 14 | 16 |
| Head | Linear (768 -> 256 -> 1) | LayerNorm + Linear (1024 -> 256 -> 1) |
| Source | torch.hub (facebookresearch/dinov2) | HuggingFace (facebook/dinov3-vitl16-pretrain-lvd1689m) |
| Input size | 384x384 | 384x384 |

Both models use the same two-stage training protocol:
1. **Linear probe** (20 epochs): Backbone frozen, train head only at LR=1e-3
2. **Fine-tune** (80 epochs): Full model unfrozen with differential LR (backbone: 1e-5, head: 1e-3)

## Results

### Cross-Validation Performance (5-Fold)

| Model | AUPRC | AUROC | F1 (optimal) |
|-------|-------|-------|--------------|
| A - EfficientNetV2-S | 0.975 +/- 0.010 | 0.991 +/- 0.004 | 0.909 +/- 0.013 |
| B - YOLO + CNN-LAB | 0.974 +/- 0.012 | 0.992 +/- 0.003 | 0.905 +/- 0.015 |
| **C - DINOv2-Base** | **0.994 +/- 0.003** | **0.999 +/- 0.001** | **0.952 +/- 0.008** |
| D - DINOv3-Large | 0.992 +/- 0.002 | 0.998 +/- 0.001 | 0.948 +/- 0.006 |

### Model D Per-Fold Breakdown

| Fold | AUPRC | AUROC |
|------|-------|-------|
| 0 | 0.9891 | 0.9975 |
| 1 | 0.9917 | 0.9982 |
| 2 | 0.9942 | 0.9988 |
| 3 | 0.9925 | 0.9984 |
| 4 | 0.9901 | 0.9978 |
| **Mean** | **0.9915** | **0.9981** |
| **Std** | **0.0020** | **0.0005** |

Model D's variance is slightly lower than Model C (std 0.002 vs 0.003), suggesting more stable training, but the mean AUPRC is 0.002 points lower.

## Calibration

Temperature scaling was applied using validation predictions from all 5 folds.

| Metric | Model C | Model D |
|--------|---------|---------|
| Temperature (T) | 1.186 | 1.296 |
| ECE (before) | 0.0120 | 0.0148 |
| ECE (after) | 0.0085 | 0.0107 |
| Threshold @ 95% recall | 0.2184 | 0.2572 |

Both models benefit from temperature scaling (T > 1 indicates slight overconfidence). Model C achieves better calibration (lower ECE after scaling) and a lower decision threshold to reach 95% recall.

## Streaming Inference Validation

Model D was validated on the full iNaturalist Anura corpus via streaming inference.

| Metric | Value |
|--------|-------|
| Total photos scored | 147,048 |
| Flagged at 0.9 threshold | ~2,060 (1.4%) |
| Flagged at 0.5 threshold | ~4,200 (2.9%) |
| Inference time per image | ~45ms (A100 GPU) |

For comparison, Model C flagged ~1,800 photos (1.2%) at the same 0.9 threshold on a comparable streaming run, indicating Model D is slightly less conservative (more false positives at the same threshold).

### Score Distribution

The vast majority of images score near 0.0 (clearly not axanthic). The distribution shows a clean separation between negative and flagged populations with a sparse transition region between 0.1 and 0.5.

## Conclusion

**Model C (DINOv2-Base) remains the best model for production use.**

Model D's larger architecture and newer pretraining did not translate to improved axanthism detection. The likely reasons:

1. **Task saturation**: At AUPRC > 0.99, the task is nearly solved. The remaining errors are ambiguous cases (poor lighting, unusual angles) where more parameters don't help.
2. **Dataset size**: With ~10k training images, a 300M parameter model may be slightly overparameterized compared to 86M, despite regularization.
3. **Domain gap**: DINOv3's additional training data (internet images at large) doesn't specifically improve on the ecological image domain where DINOv2 already performs well.

## Recommendations

1. **Use Model C for production deployment** -- better AUPRC, better calibration, 3.5x smaller, and faster inference.
2. **Threshold selection**: Use the calibrated threshold from `calibration.json` (0.2184 for 95% recall). For high-precision screening where manual review follows, a threshold of 0.9 keeps the flagged set manageable (~1.2% of corpus).
3. **Model D as a complementary signal**: In a future ensemble, Model D's independent errors could improve recall at fixed precision, but the marginal gain is likely small given the already high performance.
4. **Focus further work on data quality** -- curating hard negatives and expanding positives will likely yield more improvement than architecture changes at this point.
