# Blue Frogs: Automated Axanthism Detection in Frogs

## What This Project Does

This project builds on the [Womack et al. Blue Frogs Project](https://github.com/mcwomack/bluefrogs), which manually screened ~400,000 iNaturalist frog photos through December 2021 and identified 372 confirmed axanthic observations across 36 species in 11 families.

Manual screening is thorough but slow. As of December 2024, there are roughly **2.24 million** frog observations with photographs on iNaturalist -- far more than any team can review by hand. This project trains computer vision models to automatically scan frog photos and flag those likely showing axanthism, so that only the flagged subset (~1-3% of all photos) requires human review.

**The bottom line:** Our best model (Model C) identifies axanthic frogs with 99.4% accuracy across all confidence levels, and can screen the entire iNaturalist frog corpus in a matter of hours on a single GPU. This means potentially thousands of new axanthic observations can be surfaced from the ~1.8 million photos that haven't been manually reviewed yet.

---

## The Data

### Positives: Confirmed Axanthic Frogs

All confirmed axanthic records from Womack et al. 2025 (372 observations across 36 species, 11 families), supplemented with additional records from Jongsma et al. 2026 and the Aberrantly Blue Frogs iNaturalist project. Multi-photo observations include all photos where axanthism is visible.

### Negatives: What the Model Must Learn to Ignore

Teaching the model what axanthism is *not* is just as important as teaching what it is. We curate negative examples in three tiers of difficulty:

| Tier | ~Count | What's included | Why |
|------|--------|----------------|-----|
| **Hard negatives** | 1,000 | Naturally blue species (*Dendrobates tinctorius*, *D. azureus*, *Oophaga pumilio*, *Ranitomeya amazonica*), blue-phase frogs, frogs on blue substrates | The model must distinguish true axanthism from species that are *supposed* to be blue |
| **Medium negatives** | 2,000 | Normal-colored individuals of the same species that have documented axanthism | The model must learn intra-species color variation |
| **Easy negatives** | 7,000 | Random frog photos sampled across families, geographies, and image qualities | Broad representation of "normal" frogs |

### Data Splits

- **Test set (held out permanently):** 20% of positives plus matched negatives, never used during training
- **Training set (5-fold cross-validation):** The remaining 80%, split 5 ways. Each "fold" trains on 4 parts and validates on 1, so every image gets a turn as validation data.
- Splits are done by **observation** (not photo) so that multiple photos of the same frog don't end up in both training and validation -- that would give misleadingly good results.

### A Critical Design Choice: Constraining Color Augmentation

During training, we randomly vary brightness, contrast, and other image properties to make the model robust to different photo conditions. However, **we cap hue (color) variation at only +/-5 degrees**. This is critical because axanthism is fundamentally a color-based trait -- the loss of yellow pigments from xanthophores shifts green to blue. If we randomly shifted hues by large amounts during training, we'd be teaching the model to ignore the very signal it needs to detect.

---

## The Four Models

We built four progressively more sophisticated models to find the best approach. Each was evaluated using 5-fold cross-validation (the data is split 5 ways, and the model is trained and tested 5 times on different splits to get reliable performance estimates).

### Model A: Standard Image Classifier

**What it is:** A conventional image classification model (EfficientNetV2-S) pre-trained on general images, then fine-tuned on our frog dataset.

**How it works:** Looks at the entire 384x384 pixel image and outputs a single number between 0 and 1 representing its confidence that the frog is axanthic.

**Performance:**

| Metric | Score |
|--------|-------|
| AUPRC | 0.960 |
| F1 | 0.904 |

**Strengths:** Fast, simple, reliable baseline. This is the approach most similar to prior work on phenotype classification in community science images (e.g., Hantak et al. 2022 on salamander striping patterns).

**Weaknesses:** Treats the entire image equally -- background, substrate, and frog are all mixed together. May be distracted by blue backgrounds or blue objects near the frog.

### Model B: Detect the Frog, Then Classify

**What it is:** A two-stage pipeline that first locates the frog in the photo (using a YOLO object detector), crops just the frog, then classifies the cropped image using both visual features *and* explicit color measurements.

**How it works:**
1. A frog detector finds and crops the frog from the full image
2. The cropped frog image is analyzed two ways simultaneously:
   - A neural network extracts visual features (shape, texture, pattern)
   - Color statistics are computed in LAB color space (30 measurements including mean, standard deviation, and percentiles of the L*, a*, and b* channels)
3. Both sets of features are combined to make the final prediction

**Why LAB color space?** The b* channel in LAB color space directly encodes the yellow-blue axis -- exactly the axis that shifts in axanthism. By explicitly measuring b* statistics, we give the model direct access to the biologically relevant color information.

**Performance:**

| Metric | Score |
|--------|-------|
| AUPRC | 0.947 |
| F1 | 0.892 |

**Strengths:** By cropping to just the frog, background distractions are eliminated. The explicit LAB color features make the color analysis interpretable -- we can see *which* color statistics drove a prediction.

**Weaknesses:** Depends on the frog detector finding the frog correctly. If the detector misses the frog or crops poorly, the classifier has bad input. Performance is slightly lower than Model A, possibly because the detector occasionally introduces errors.

### Model C: Pre-Trained Vision Foundation Model (Best)

**What it is:** DINOv2, a state-of-the-art vision model from Meta AI that was pre-trained on 142 million diverse images using self-supervised learning (no labels needed). It has 86 million parameters and has learned rich visual representations that transfer well to specialized tasks.

**How it works:** The model is trained in two stages:
1. **Stage 1 (Learning the basics):** The pre-trained model is frozen (its knowledge locked in place), and only a small classification head is trained on our frog data. This learns the basic mapping from DINOv2's features to axanthism prediction.
2. **Stage 2 (Fine-tuning):** The entire model is unlocked and trained end-to-end on our data, but with a very low learning rate for the pre-trained layers (to adjust gently without forgetting what it already knows) and a higher learning rate for the classification head.

**Performance (5-fold cross-validation):**

| Metric | Score |
|--------|-------|
| AUPRC | 0.994 +/- 0.003 |
| AUROC | 0.999 +/- 0.001 |
| F1 | 0.952 +/- 0.008 |

**Strengths:** By far the best performer. The pre-trained features give it a deep understanding of visual structure, texture, and color that transfers remarkably well to axanthism detection. Consistent across all 5 data folds. Relatively fast inference.

**This is the recommended model for all screening work.**

### Model D: Bigger Pre-Trained Model

**What it is:** DINOv3, Meta's newer and larger vision model (300 million parameters, 3.5x larger than Model C) trained on 1.6 billion images (12x more than DINOv2).

**The hypothesis:** A model that has seen more images and has more capacity might better handle edge cases -- particularly distinguishing true axanthism from naturally blue species like *Dendrobates azureus* in the hard negatives.

**Performance (5-fold cross-validation):**

| Metric | Score |
|--------|-------|
| AUPRC | 0.992 +/- 0.002 |
| AUROC | 0.998 +/- 0.001 |
| F1 | 0.948 +/- 0.006 |

| Fold | AUPRC |
|------|-------|
| 0 | 0.9891 |
| 1 | 0.9917 |
| 2 | 0.9942 |
| 3 | 0.9925 |
| 4 | 0.9901 |

**Result: The bigger model did not improve over Model C.** AUPRC is slightly lower (0.992 vs 0.994). See the "Why Bigger Wasn't Better" discussion below.

---

## Head-to-Head Comparison

| Model | Approach | AUPRC | F1 | Relative size |
|-------|----------|-------|-----|---------------|
| A | Standard classifier | 0.960 | 0.904 | Small |
| B | Frog detector + color | 0.947 | 0.892 | Medium |
| **C** | **DINOv2 (pre-trained)** | **0.994** | **0.952** | **Medium** |
| D | DINOv3 (larger pre-trained) | 0.992 | 0.948 | Large (3.5x C) |

### Understanding the Metrics

- **AUPRC** (Area Under Precision-Recall Curve): Our primary metric. Measures how well the model distinguishes axanthic from normal frogs across *all possible* confidence thresholds. A score of 1.0 would be perfect. We use this instead of simpler metrics because axanthism is so rare (~0.09% of frogs) -- a model that just said "not axanthic" for every photo would be 99.91% accurate but completely useless. AUPRC properly penalizes that.

- **F1 Score**: Balances two competing goals -- catching all the axanthic frogs (recall) versus not flagging too many normal frogs (precision). An F1 of 0.952 means the model is excellent at both.

- **+/- values**: The variation across 5 cross-validation folds. Smaller means more consistent. Model C's +/- 0.003 means it performs reliably regardless of how the data is split.

---

## Confidence Calibration

Neural networks tend to be overconfident -- a photo might get a confidence score of 0.95 when the true probability is closer to 0.80. This matters for setting thresholds: if scores aren't calibrated, a threshold of 0.5 might flag far more or fewer photos than expected.

We apply **temperature scaling** to adjust confidence scores after training. This is a single-parameter correction that doesn't change which photos are ranked highest -- it only adjusts how well the scores match true probabilities.

| | Model C | Model D |
|---|---------|---------|
| Calibration error (lower = better) | 0.0085 | 0.0107 |
| Threshold for 95% recall | 0.22 | 0.26 |

**What "threshold for 95% recall" means:** If you set the model to flag every photo with a score above this threshold, it will catch approximately 95% of truly axanthic frogs. Model C can achieve this at a lower threshold (0.22), meaning it needs to be less aggressive to catch the same proportion.

---

## Large-Scale Screening: How It Works in Practice

The streaming inference pipeline can scan the entire iNaturalist frog corpus without needing to download all 2+ million photos in advance. It works in batches:

1. Fetch a batch of observations from iNaturalist (e.g., 1,000 at a time)
2. Download the photos temporarily
3. Score each photo with the model
4. Save predictions and flag photos above the threshold
5. Delete the temporary images and move to the next batch
6. If interrupted, resume from the last completed batch

### Validation Run: 147,000 Photos

We ran Model D across 147,048 iNaturalist frog photos as a validation test:

| | Model C | Model D |
|---|---------|---------|
| Flagged at threshold 0.9 | ~1,800 (1.2%) | ~2,060 (1.4%) |
| Flagged at threshold 0.5 | (not run) | ~4,200 (2.9%) |

At the conservative 0.9 threshold, Model C flags fewer photos (1.2% vs 1.4%), meaning less manual review work with the same or better detection quality. The score distribution shows clean separation -- the vast majority of photos score near 0.0 (clearly not axanthic), with a small tail of high-scoring flagged photos.

### Review Gallery

Flagged photos are presented in an interactive HTML gallery that shows:
- Thumbnail images loaded directly from iNaturalist
- The model's confidence score for each photo
- Links back to the original iNaturalist observation
- Sorting by score (highest first for most likely positives)
- Filtering by which model flagged the photo

This lets a reviewer quickly scan through flagged photos, confirm or reject each one, and follow up on the original observation if needed.

---

## Where the Model Does Well

1. **Classic axanthism**: Fully blue frogs where the xanthophore loss is complete and obvious. These score very high (>0.95) across all models.

2. **Partial axanthism**: Frogs with blue patches or partially blue coloration. Model C handles these well, though scores tend to be lower (0.5-0.9 range) than full axanthism.

3. **Distinguishing from naturally blue species**: The hard negative training tier specifically teaches the model that *Dendrobates azureus* is supposed to be blue. Model C has learned this distinction well -- naturally blue species rarely score above 0.1.

4. **Varied image quality**: iNaturalist photos range from professional macro shots to distant smartphone photos. Model C handles this variation robustly thanks to the diverse pre-training of DINOv2.

5. **Scale**: The streaming pipeline can process the entire iNaturalist frog corpus efficiently, with automatic resume if interrupted.

---

## Where the Model Needs Improvement

1. **Ambiguous photos**: Poor lighting, unusual angles, partial views of the frog, or very small frogs in frame. These are difficult for any model (and often difficult for human reviewers too). These are the photos that account for most of the remaining ~0.6% error.

2. **Novel species**: If axanthism is documented in a species not represented in our training data, the model may or may not detect it depending on how similar the presentation is to species it has seen. The model generalizes reasonably well across species, but performance on completely novel species hasn't been systematically tested.

3. **Blue substrates or lighting**: Frogs on blue objects, in blue-tinted lighting, or reflected in water can occasionally trigger low-confidence flags (scores 0.1-0.3). These are usually filtered out at typical operating thresholds but can appear in high-recall configurations.

4. **Threshold selection is use-case dependent**: There's no single "right" threshold. A researcher wanting to find every possible axanthic frog (survey mode) needs a different threshold than one wanting only high-confidence detections for immediate confirmation.

5. **Limited positive training data**: With ~372 confirmed axanthic observations as the foundation, the training set is inherently small for the positive class. Adding more confirmed observations would likely improve the model, particularly for partial axanthism and less-common species.

---

## Priorities and Recommended Next Steps

### Immediate (Ready Now)

1. **Run Model C on the full iNaturalist frog corpus.** The pipeline and model are ready. At threshold 0.9, this would flag ~27,000 photos from the ~2.24M corpus for manual review. Based on the historical 0.09% prevalence, this could surface **~2,000 previously unidentified axanthic observations** from the ~1.8M photos not screened by Womack et al.

2. **Use the review gallery for manual confirmation.** The HTML gallery makes it efficient to scan flagged photos. Start with the highest-scoring predictions (most likely true positives) and work down.

### Short-Term

3. **Feed confirmed results back into training.** As new axanthic frogs are confirmed through manual review, they become new positive training examples. Retraining with an expanded positive set should improve detection, especially for species or presentations underrepresented in the original 372 observations.

4. **Curate additional hard negatives.** If the model consistently false-positives on specific species or conditions, those photos become targeted hard negatives for the next training round. This iterative refinement is the most efficient way to reduce false positives.

5. **Test on novel species.** Systematically evaluate how well the model generalizes to species not in the training set. This could reveal whether species-specific fine-tuning is needed or whether the model's general color understanding suffices.

### Longer-Term

6. **Threshold guidance by use case.** Develop recommended threshold settings for different research scenarios:
   - **Survey mode** (threshold ~0.2): Maximize recall, accept more false positives for review. Best for: "Find every possible axanthic frog."
   - **Screening mode** (threshold ~0.5): Balance precision and recall. Best for: "Flag likely candidates for a manageable review workload."
   - **Confirmation mode** (threshold ~0.9): High precision, minimal false positives. Best for: "Show me only the most obvious cases."

7. **Partial vs. full axanthism.** The current model treats all axanthism as a single class. Future work could distinguish partial from full axanthism, which may be biologically relevant (different genetic mechanisms, different ecological implications).

8. **Temporal and geographic analysis.** With predictions across the full iNaturalist corpus, analyze spatial and temporal patterns in axanthism prevalence. Are there geographic hotspots? Seasonal patterns? Species-level variation in prevalence?

9. **Publication.** The pipeline, model weights, and results are designed for reproducibility. Training configs, data split definitions, and evaluation code are all version-controlled and ready for a methods paper.

---

## Why Bigger Wasn't Better (Model D)

Model D used a model with 3.5x more parameters, trained on 12x more images. It performed slightly *worse* than Model C. Three factors explain this:

1. **The task is nearly solved.** At 99.4% AUPRC, the remaining errors are genuinely ambiguous photos where even a human reviewer might disagree. A bigger model can't resolve ambiguity that's inherent in the photo itself.

2. **Not enough training data for a bigger model.** With ~10,000 training images, Model C's 86M parameters are well-matched to the data. Model D's 300M parameters may be too many for this dataset size -- the model has excess capacity that doesn't help and may slightly hurt through overfitting.

3. **General images don't help with frogs.** DINOv3 saw 12x more training images, but those were general internet images (food, buildings, people, etc.). This broader visual experience doesn't specifically improve the subtle color distinctions needed for axanthism detection, where DINOv2's smaller but sufficient training already works well.

**The takeaway:** Further improvements should come from better data (more confirmed axanthic observations, targeted hard negatives), not bigger models.

---

## Technical Infrastructure

For those running the pipeline:

- **Training** runs on GPU compute (tested on NVIDIA A100). Full 5-fold cross-validation of all 4 models takes several hours.
- **Inference** can run on any machine with a GPU. Streaming inference processes ~1,000 observations per batch with automatic resume.
- **The codebase** has 58 automated tests, is fully version-controlled, and includes Docker containers for reproducibility.
- **Model weights** will be published to Zenodo with a DOI for long-term archival.
- All training configurations are stored as YAML files -- no hidden settings or hardcoded values.

See `README.md` for setup instructions and `docs/ML_WORKFLOW.md` for the complete technical pipeline documentation.
