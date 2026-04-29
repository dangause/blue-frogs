"""Generate a dataset card (DATASET_CARD.md) for the axanthism classifier.

Auto-generates documentation including:
- Data sources and collection methodology
- Train/val/test split statistics
- Negative tier breakdown
- Known biases and limitations

Usage:
    python scripts/generate_dataset_card.py \
        --labels-file data/labels.json \
        --splits-file data/splits/train_test_split.json \
        --output docs/DATASET_CARD.md
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from blue_frogs.config import (
    DATA_DIR,
    WOMACK_GITHUB_URL,
    ANURA_TAXON_ID,
    INAT_API_BASE,
    RANDOM_SEED,
    N_FOLDS,
    TEST_FRACTION,
)
from blue_frogs.data.curation import HARD_NEGATIVE_SPECIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_CARD_TEMPLATE = '''# Dataset Card: Blue Frogs Axanthism Detection

## Dataset Description

### Overview

This dataset supports training and evaluation of machine learning models for detecting axanthism (blue coloration mutation) in wild frogs. Axanthism is a genetic mutation that prevents yellow pigment (xanthophores) production, resulting in frogs appearing blue instead of green.

### Data Sources

1. **Womack et al. (2023) Verified Records**
   - Source: [{womack_url}]({womack_url})
   - Contains verified axanthic frog observations with taxonomic and geographic metadata
   - Used as positive examples (ground truth axanthic)

2. **iNaturalist Research-Grade Observations**
   - API: {inat_api}
   - Taxon ID: {anura_taxon_id} (Order Anura - frogs and toads)
   - Filtered for research-grade observations with photos
   - Used for negative examples (presumed normal coloration)

### Collection Methodology

#### Positive Examples (Axanthic)
- Sourced from peer-reviewed compilation by Womack et al.
- Each record manually verified by herpetologists
- Photos downloaded from iNaturalist using observation IDs

#### Negative Examples (Normal)
Stratified sampling with three difficulty tiers:

| Tier | Description | Count |
|------|-------------|-------|
| **Hard** | Naturally blue/grey species that could confuse the model | {n_hard} |
| **Medium** | Same species as positives but normal coloration | {n_medium} |
| **Easy** | Random frog species | {n_easy} |

Hard negative species include:
{hard_negative_list}

---

## Dataset Statistics

### Overall Composition

| Split | Total | Positive | Negative | Pos Ratio |
|-------|-------|----------|----------|-----------|
| Train | {n_train} | {n_train_pos} | {n_train_neg} | {train_pos_ratio:.1%} |
| Test  | {n_test} | {n_test_pos} | {n_test_neg} | {test_pos_ratio:.1%} |

### Cross-Validation

- Number of folds: {n_folds}
- Stratified by label to maintain class balance
- Random seed: {random_seed}

### Test Set

- Held-out fraction: {test_fraction:.0%}
- Stratified by label
- Used for final model evaluation only

---

## Known Biases and Limitations

### Geographic Bias
- Positive examples heavily weighted toward North America (where axanthism is most documented)
- May underperform on species from underrepresented regions

### Temporal Bias
- iNaturalist data skews toward recent observations (2015-present)
- Historical records may have different image quality characteristics

### Species Coverage
- Not all frog families equally represented
- Some rare species have few or no examples

### Image Quality Variation
- Mix of professional DSLR and smartphone photos
- Varying lighting conditions, angles, and resolutions
- Some images contain multiple frogs or partial views

### Label Noise
- Negative examples assumed normal (not verified non-axanthic)
- Some edge cases may exist (partial axanthism, color variation)

---

## Recommended Uses

### Intended Use Cases
- Automated screening of iNaturalist observations for potential axanthism
- Supporting biodiversity monitoring and conservation research
- Educational demonstrations of rare phenotype detection

### Out-of-Scope Uses
- Definitive diagnosis of axanthism (always requires expert verification)
- Detection in non-anuran species
- Real-time field identification (model optimized for post-hoc screening)

---

## Maintenance

### Version
Generated: {generation_date}

### Updates
- Dataset should be regenerated when new verified axanthic records become available
- Consider periodic refresh of negative examples from iNaturalist

### Contact
For questions about this dataset, see the project repository.

---

## Citation

If using this dataset, please cite:

```
Womack, M. C., et al. (2023). Blue frogs: A database of anuran color anomalies.
```

And the iNaturalist data source:

```
iNaturalist contributors. iNaturalist Research-grade Observations.
Available at: https://www.inaturalist.org
```
'''


def load_splits_stats(splits_file: Path, labels_file: Path) -> dict:
    """Load and compute statistics from splits and labels files."""
    stats = {
        "n_train": 0,
        "n_train_pos": 0,
        "n_train_neg": 0,
        "train_pos_ratio": 0.0,
        "n_test": 0,
        "n_test_pos": 0,
        "n_test_neg": 0,
        "test_pos_ratio": 0.0,
    }

    if not splits_file.exists():
        logger.warning(f"Splits file not found: {splits_file}")
        return stats

    if not labels_file.exists():
        logger.warning(f"Labels file not found: {labels_file}")
        return stats

    with open(labels_file) as f:
        all_labels = json.load(f)

    with open(splits_file) as f:
        splits = json.load(f)

    train_indices = splits.get("train_indices", [])
    test_indices = splits.get("test_indices", [])

    train_labels = [all_labels[i]["label"] for i in train_indices]
    test_labels = [all_labels[i]["label"] for i in test_indices]

    stats["n_train"] = len(train_labels)
    stats["n_train_pos"] = sum(train_labels)
    stats["n_train_neg"] = stats["n_train"] - stats["n_train_pos"]
    stats["train_pos_ratio"] = stats["n_train_pos"] / stats["n_train"] if stats["n_train"] > 0 else 0

    stats["n_test"] = len(test_labels)
    stats["n_test_pos"] = sum(test_labels)
    stats["n_test_neg"] = stats["n_test"] - stats["n_test_pos"]
    stats["test_pos_ratio"] = stats["n_test_pos"] / stats["n_test"] if stats["n_test"] > 0 else 0

    return stats


def estimate_negative_tiers(labels_file: Path) -> dict:
    """Estimate negative tier counts from labels file."""
    # Default estimates based on curation.py defaults
    tiers = {
        "n_hard": 1000,
        "n_medium": 2000,
        "n_easy": 7000,
    }

    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)

        # If we have tier information in the labels
        tier_counts = {"hard": 0, "medium": 0, "easy": 0}
        for item in labels:
            if item.get("label") == 0:
                tier = item.get("negative_tier", "unknown")
                if tier in tier_counts:
                    tier_counts[tier] += 1

        if sum(tier_counts.values()) > 0:
            tiers["n_hard"] = tier_counts["hard"]
            tiers["n_medium"] = tier_counts["medium"]
            tiers["n_easy"] = tier_counts["easy"]

    return tiers


def main():
    parser = argparse.ArgumentParser(description="Generate dataset card")
    parser.add_argument("--labels-file", type=Path, default=DATA_DIR / "labels.json",
                        help="Path to labels.json")
    parser.add_argument("--splits-file", type=Path, default=DATA_DIR / "splits" / "train_test_split.json",
                        help="Path to train_test_split.json")
    parser.add_argument("--output", type=Path, default=Path("docs") / "DATASET_CARD.md",
                        help="Output path for DATASET_CARD.md")
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Gather statistics
    stats = load_splits_stats(args.splits_file, args.labels_file)
    tiers = estimate_negative_tiers(args.labels_file)

    # Format hard negative species list
    hard_negative_list = "\n".join(f"- *{species}*" for species in HARD_NEGATIVE_SPECIES)

    # Generate the card
    card_content = DATASET_CARD_TEMPLATE.format(
        womack_url=WOMACK_GITHUB_URL,
        inat_api=INAT_API_BASE,
        anura_taxon_id=ANURA_TAXON_ID,
        hard_negative_list=hard_negative_list,
        n_hard=tiers["n_hard"],
        n_medium=tiers["n_medium"],
        n_easy=tiers["n_easy"],
        n_train=stats["n_train"],
        n_train_pos=stats["n_train_pos"],
        n_train_neg=stats["n_train_neg"],
        train_pos_ratio=stats["train_pos_ratio"],
        n_test=stats["n_test"],
        n_test_pos=stats["n_test_pos"],
        n_test_neg=stats["n_test_neg"],
        test_pos_ratio=stats["test_pos_ratio"],
        n_folds=N_FOLDS,
        random_seed=RANDOM_SEED,
        test_fraction=TEST_FRACTION,
        generation_date=datetime.utcnow().strftime("%Y-%m-%d"),
    )

    # Write the card
    with open(args.output, "w") as f:
        f.write(card_content)

    logger.info(f"Dataset card saved to {args.output}")


if __name__ == "__main__":
    main()
