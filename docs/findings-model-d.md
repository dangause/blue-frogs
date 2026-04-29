# Model D Findings: DINOv3 (Larger Vision Model)

## Summary

We tested whether a bigger, newer AI vision model (DINOv3, 300 million parameters) could improve on our current best model (Model C, DINOv2, 86 million parameters) for detecting axanthism in frog photos. **It did not.** Model C remains the recommended model for screening iNaturalist images.

## Why We Tried a Bigger Model

Model C already performs very well -- it correctly identifies axanthic frogs about 99.4% of the time when measured across all confidence thresholds (AUPRC = 0.994). But the original goal of this branch was to reduce false positives: photos of normal frogs that the model incorrectly flags as potentially axanthic.

DINOv3 is Meta's newer vision model, trained on 12x more images (1.6 billion vs 142 million) and 3.5x larger than DINOv2. The hypothesis was that its richer "understanding" of images might help it better distinguish true axanthism from naturally blue species like *Dendrobates azureus* or unusual lighting conditions.

## How the Models Compare

### What the metrics mean

- **AUPRC** (Area Under Precision-Recall Curve): Measures overall detection quality across all possible confidence thresholds. Higher is better. 1.0 would be perfect. This is our primary metric because axanthism is so rare (~0.09% of frogs).
- **AUROC**: Similar to AUPRC but less sensitive to class imbalance. Included for comparison with other studies.
- **F1**: Balances the trade-off between catching all axanthic frogs (recall) and not flagging too many normal frogs (precision). Higher is better.
- **+/-**: Variation across the 5 cross-validation folds (different random splits of training data). Lower variation means more consistent results.

### Results (5-fold cross-validation)

| Model | What it is | AUPRC | F1 |
|-------|-----------|-------|-----|
| A | Standard image classifier | 0.975 +/- 0.010 | 0.909 +/- 0.013 |
| B | Frog detector + color analysis | 0.974 +/- 0.012 | 0.905 +/- 0.015 |
| **C** | **DINOv2 (86M params) -- Best** | **0.994 +/- 0.003** | **0.952 +/- 0.008** |
| D | DINOv3 (300M params) | 0.992 +/- 0.002 | 0.948 +/- 0.006 |

Model D scores slightly lower than Model C on both metrics. The difference is small (0.994 vs 0.992 AUPRC) but consistent across all 5 data splits.

### Model D results by fold

Each fold uses a different 80/20 split of the training data, so this shows how stable the results are:

| Fold | AUPRC |
|------|-------|
| 0 | 0.9891 |
| 1 | 0.9917 |
| 2 | 0.9942 |
| 3 | 0.9925 |
| 4 | 0.9901 |
| **Average** | **0.9915 +/- 0.0020** |

## Calibration (Confidence Tuning)

Neural networks often output overconfident predictions -- a photo might get a score of 0.95 when the true probability of axanthism is closer to 0.80. Calibration adjusts for this so the scores better reflect actual probabilities.

After calibration:

| | Model C | Model D |
|---|---------|---------|
| Calibration error (lower = better) | 0.0085 | 0.0107 |
| Score threshold to catch 95% of axanthic frogs | 0.22 | 0.26 |

**What this means in practice:** To catch 95% of truly axanthic frogs, Model C only needs to flag photos scoring above 0.22, while Model D needs a higher cutoff of 0.26. Model C's lower threshold means it casts a slightly wider net while still maintaining good precision.

## Large-Scale Validation on iNaturalist

We ran Model D across 147,048 iNaturalist frog photos to see how it performs at scale:

| | Model C | Model D |
|---|---------|---------|
| Photos scored | ~147k | 147,048 |
| Flagged at high-confidence threshold (0.9) | ~1,800 (1.2%) | ~2,060 (1.4%) |

Model D flags about 15% more photos than Model C at the same confidence threshold, meaning more false positives to manually review.

## Why the Bigger Model Didn't Help

1. **The task is nearly solved.** At 99.4% AUPRC, the remaining errors are genuinely ambiguous photos (bad lighting, unusual angles, partial views) where even a larger model can't reliably decide. More parameters don't help with genuinely ambiguous data.

2. **Not enough training data for a bigger model.** With ~10,000 training images, the 300M-parameter model may be too large relative to the available data. The 86M-parameter Model C is a better fit for this dataset size.

3. **General images don't help with frogs.** DINOv3's extra training data is mostly general internet images. This doesn't specifically improve recognition of the subtle color differences that distinguish axanthism from natural blue coloration in species like *Dendrobates*.

## Recommendations

1. **Use Model C for all screening.** It's more accurate, better calibrated, 3.5x smaller (faster to run), and flags fewer false positives.

2. **Threshold guidance for manual review:**
   - **Threshold 0.9** -- flags ~1.2% of corpus. Good for manageable review batches. Very few false negatives among high-scoring photos.
   - **Threshold 0.22** (calibrated) -- catches ~95% of axanthic frogs but flags more photos for review.
   - Choose based on whether you prioritize completeness (lower threshold) or efficiency of manual review (higher threshold).

3. **Next steps for improvement** should focus on data, not model architecture:
   - Add more confirmed axanthic observations to the training set
   - Curate additional hard negatives (naturally blue species, unusual lighting)
   - These will likely improve detection more than trying yet another model architecture

## Technical Details

For ML practitioners, additional architecture details are in:
- `configs/model_d.yaml` -- training hyperparameters
- `src/blue_frogs/models/model_d.py` -- model implementation
- `docs/ML_WORKFLOW.md` -- full pipeline documentation
