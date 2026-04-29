# Axanthism Classifier Design Document

**Date:** 2026-02-26
**Team:** CAS Blue Frogs Project (working with Rayna C. Bell)
**Goal:** Train a CV model to classify axanthism (blue coloration) in frogs from iNaturalist images, compare three model architectures, batch-score the full iNaturalist frog corpus, and publish results.

## Scientific Context

Axanthism -- the absence or strong reduction of yellow pigments from xanthophores -- causes normally green frogs to appear blue. The Blue Frogs Project (Womack et al. 2025, J. Herpetology) manually screened ~400,000 of ~900,000 iNaturalist frog photos through Dec 2021, identifying 372 axanthic observations across 36 species in 11 families. As of Dec 2024, there are ~2.24 million frog observations with photographs on iNaturalist, far exceeding what can be manually screened.

Hantak et al. (2022, iScience) demonstrated CV-based phenotype classification on community science images, achieving ~98% accuracy classifying striped/unstriped salamanders using an EfficientNet-b4 ensemble trained on ~4,000 volunteer-scored images.

This project extends that approach to axanthism detection, comparing three model families and deploying the best to score the full iNaturalist frog corpus at scale.

### Key references

- Womack et al. 2025 -- Blue Frogs Project dataset (J. Herpetology 59(3):1-9)
- Hantak et al. 2022 -- CV for species color pattern variation (iScience 25:104784)
- Jongsma et al. 2026 -- Updated axanthism checklist (Herpetology Notes 19:31-37)
- Barve et al. 2020 -- Best practices for scoring citizen science photos (Appl. Plant Sci. 8(1):e11315)

## Overall Architecture

Four major phases:

1. **Data Pipeline** -- Download images, curate training/validation/test sets
2. **Model Development** -- Train and evaluate three model families under identical conditions
3. **Batch Inference** -- Score the full iNaturalist frog corpus with the best model(s)
4. **Analysis & Reporting** -- Generate figures, statistics, and reproducibility artifacts

### Classification task

- **Primary:** Binary classification (axanthic vs. normal)
- **Future refinement:** Partial vs. full axanthism as a second stage

### Key constraints

- ~372 positives out of ~400k screened images (0.09% positive rate)
- Data source: iNaturalist observation IDs with labels (images downloaded via API)
- Compute: HPC cluster with multi-GPU nodes
- Science-grade: reproducibility, rigorous evaluation, publishable methods

## Phase 1: Data Pipeline

### 1a. Image Acquisition

- Download images via iNaturalist API using observation IDs from Womack et al. dataset
- Download "medium" resolution (1024px longest side) per image; store original URL for full-res if needed
- Store metadata: observation_id, species, coordinates, date, observer, quality_grade, photo_count
- For multi-photo observations, download all photos
- Target: full screened set (~400k) + complete current iNat frog corpus for inference

### 1b. Training Set Curation

**Positives (~400-500 images):**
- All confirmed axanthic observations from Womack et al.
- Additional records from Jongsma et al. 2026 and ongoing Aberrantly Blue Frogs iNaturalist project (up to defined data cutoff)
- Multi-photo observations: include all photos where axanthism is visible
- Metadata includes partial vs. full axanthism labels for future refinement

**Negatives (~10,000 images) -- tiered curation:**

| Tier | Description | ~Count | Purpose |
|------|-------------|--------|---------|
| Hard negatives | Non-axanthic frogs with blue/grey tones (e.g., *D. azureus*, blue-phase *D. auratus*, frogs on blue substrates) | ~1,000 | Prevent false positives on naturally blue or blue-background frogs |
| Medium negatives | Normal individuals of species where axanthism has been documented | ~2,000 | Teach species-level color variation |
| Easy negatives | Random sample stratified across frog families, geographies, image quality | ~7,000 | Broad representation of "normal" |

### 1c. Data Splits

- **Test set (held out permanently):** 20% of positives (~80-100 images) + matched negatives from each tier, stratified by species
- **Development set (5-fold cross-validation):** Remaining 80% of positives + negatives
- Split by **observation** (not photo) to prevent data leakage
- Split by **genus** where possible to test taxonomic generalization (fallback: random stratified if too few positives per fold)

### 1d. Preprocessing & Augmentation

**Preprocessing (all models):**
- Resize to 384x384
- Normalize with model-specific mean/std

**Augmentation (training only):**
- Random horizontal flip
- Random rotation (+/- 30 degrees)
- Random resized crop (scale 0.7-1.0)
- Color jitter (brightness, contrast, saturation -- conservative on hue)
- Random affine transforms
- CutMix/MixUp with 0.2 probability
- **Hue augmentation capped at +/- 5 degrees** -- axanthism is a color-based trait; we must not augment green frogs to look blue

## Phase 2: Model Development

### Shared evaluation protocol

**Cross-validation metrics (mean +/- std across folds):**
- AUPRC (primary metric -- most informative under extreme class imbalance)
- AUROC
- F1 at optimal threshold (selected per fold on validation set)
- Precision at 95% recall
- Recall at 95% precision

**Threshold selection:**
- Optimal F1 threshold per model
- High-recall threshold (95% recall) for batch scoring with human review
- High-precision threshold (95% precision) for conservative flagging

### Model A: Ensemble CNN (Hantak-style baseline)

**Backbone:** EfficientNetV2-S pretrained on ImageNet (modern successor to Hantak's EfficientNet-b4)

**Head:** Global average pooling -> Dropout(0.3) -> Linear(1280, 1) -> Sigmoid

**Training:**
- 5-fold stratified cross-validation
- SGD with momentum 0.9, initial lr=0.001, cosine annealing schedule
- Weighted BCE loss (weight = negatives/positives ratio, ~20:1 in curated set)
- Oversampling of positives to ~1:3 effective ratio per batch
- Batch size 32, up to 80 epochs, early stopping (patience=10 on validation AUPRC)
- Best checkpoint per fold by validation AUPRC

**Ensemble:** Average sigmoid outputs from 5 fold-best models

**Ablation:** Compare weighted BCE vs. focal loss (gamma=2)

### Model B: Two-Stage Detection + Classification

**Stage 1 -- Frog Localization:**
- Pretrained YOLOv8 or MegaDetector as frog detector
- Fine-tune on ~200-500 manually annotated bounding boxes if off-the-shelf performance is insufficient
- Output: cropped frog region, resized to 384x384

**Stage 2 -- Color Classification:**
- Learned features: EfficientNetV2-S on cropped images (same as Model A)
- Explicit color features: LAB color space histograms from cropped region. The b* channel encodes the yellow-blue axis -- directly maps to xanthophore pigment loss biology. Extract mean/median/std of b* values (~30 features)
- Fusion: Concatenate CNN features (1280-dim) + color features (~30-dim) -> Linear(1310, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1) -> Sigmoid

**Training:** Same 5-fold CV protocol as Model A, on cropped images

### Model C: Foundation Model (DINOv2 / BioCLIP)

**Backbones (test both, report best):**
- DINOv2-B/14 (ViT-Base, 86M params) -- state-of-the-art self-supervised vision transformer
- BioCLIP -- CLIP fine-tuned on TreeOfLife-10M biodiversity dataset (has seen iNaturalist frog images)

**Head:** [CLS] token -> LayerNorm -> Linear(768, 256) -> GELU -> Dropout(0.3) -> Linear(256, 1) -> Sigmoid

**Training:**
- Linear probe: freeze backbone, train head for 20 epochs (baseline for pretrained features)
- Full fine-tune: unfreeze backbone, lr=1e-5 backbone / lr=1e-3 head, AdamW, weight_decay=0.01
- Batch size 16 (ViT memory constraints)
- Same 5-fold CV and evaluation protocol

### Model comparison

| Aspect | Model A (CNN) | Model B (Two-stage) | Model C (Foundation) |
|--------|--------------|--------------------|--------------------|
| Backbone | EfficientNetV2-S | YOLO + EfficientNetV2-S | DINOv2 / BioCLIP |
| Input | Full-frame 384x384 | Cropped frog 384x384 | Full-frame 384x384 |
| Color features | Implicit only | Implicit + explicit LAB | Implicit only |
| Training complexity | Low | Medium (2 stages) | Low-medium |
| Interpretability | Grad-CAM | Grad-CAM + LAB analysis | Attention maps |
| Precedent | Hantak et al. 2022 | Novel for this domain | Emerging |

## Phase 3: Batch Inference

### Test set evaluation (once, after development is frozen)

- Apply each model's ensemble to held-out test set
- Full metrics + confusion matrices
- Bootstrap confidence intervals (1000 resamples)
- McNemar's test for pairwise model comparison

### Human validation study

- Random sample of ~500 images from batch inference results:
  - ~100 model-predicted positives (stratified across confidence levels)
  - ~400 model-predicted negatives (random sample)
- 3 team members independently label each (axanthic / normal / uncertain)
- Fleiss' kappa for inter-rater agreement
- Consensus labels to estimate real-world precision, recall, false discovery rate

### Batch scoring pipeline

1. Query iNaturalist API for all research-grade Anura observations with photos
2. Download images (medium resolution), parallelized across HPC nodes
3. Run inference with production ensemble model
4. Output: CSV (observation_id, photo_id, prediction_score, predicted_class, model_version)
5. Flag predictions above high-recall threshold for human review
6. Team reviews flagged images, confirms/rejects, records partial vs. full axanthism

**Expected volume:** ~2,000-5,000 flagged observations for human review (true positives + false positives at high-recall threshold)

**Quality controls:**
- Sanity check: confirm known Womack et al. positives are detected
- Track confidence distributions across species families for systematic bias
- Flag species with anomalously high false positive rates

### Interpretability analysis

- Grad-CAM heatmaps (Models A, B) / attention maps (Model C) for correct and incorrect predictions
- LAB b* channel distributions for true positives vs. true negatives (Model B)
- Failure case taxonomy: categorize false positives (blue background, naturally blue species, lighting artifact) and false negatives (partial axanthism, poor image quality, unusual pose)

## Phase 4: Reproducibility & Paper Artifacts

### Code repository (public GitHub)

- Environment specification (conda environment.yml or Docker container)
- Data download and preprocessing scripts
- Training scripts with configuration files (no hardcoded hyperparameters)
- Evaluation and figure generation scripts
- Inference pipeline scripts
- Fixed random seeds for all stochastic operations

### Data outputs

- Training/validation/test split definitions (observation IDs + labels, not images, per iNaturalist ToS)
- Model weights on Zenodo/FigShare with DOI
- Inference results (observation_id + prediction_score) as supplemental data

### Technology stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| Framework | PyTorch + PyTorch Lightning | Hantak et al. precedent, HPC-friendly |
| Experiment tracking | Weights & Biases | Free for academics |
| Data management | Python + iNaturalist API | Direct API access |
| Image processing | torchvision + albumentations | Flexible augmentation pipeline |
| Pretrained models | timm (PyTorch Image Models) | EfficientNetV2, DINOv2, BioCLIP available |
| Statistical analysis | scipy + statsmodels | Bootstrap CIs, McNemar's, logistic regression |
| Figures | matplotlib + seaborn | Publication-quality |

## Milestones

| Phase | Milestone | Key Deliverable |
|-------|-----------|-----------------|
| 1. Data Pipeline | Image download + training set curation | ~500 positives, ~10k negatives, splits defined |
| 2. Model A | Ensemble CNN trained and evaluated | 5-fold CV results, baseline metrics |
| 3. Model B | Two-stage pipeline trained and evaluated | Detection + classification, LAB analysis |
| 4. Model C | Foundation model fine-tuned and evaluated | DINOv2/BioCLIP results |
| 5. Comparison | Head-to-head statistical comparison | Comparison table, McNemar's tests, best model |
| 6. Inference | Full iNat corpus scored | Prediction CSV, flagged observations |
| 7. Validation | Team review of flagged predictions | Inter-rater agreement, performance estimates |
| 8. Artifacts | Code repo, model weights, figures | Reproducibility package, publication-ready figures |

Phases 2-4 can run in parallel on the HPC cluster once data is ready.
