"""Training set curation: build tiered negative set."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Species known to be naturally blue/grey (hard negatives)
HARD_NEGATIVE_SPECIES = [
    "Dendrobates tinctorius",
    "Dendrobates azureus",
    "Oophaga pumilio",
    "Ranitomeya amazonica",
]


def curate_negatives(
    all_observations_df: pd.DataFrame,
    positive_obs_ids: set[int],
    positive_species: set[str],
    n_hard: int = 1000,
    n_medium: int = 2000,
    n_easy: int = 7000,
    seed: int = 42,
) -> pd.DataFrame:
    """Select tiered negative examples from the screened observation pool.

    Args:
        all_observations_df: DataFrame of all screened observations with columns
            [observation_id, species, family, ...]
        positive_obs_ids: Set of observation IDs confirmed axanthic
        positive_species: Set of species names with documented axanthism
        n_hard: Number of hard negatives (naturally blue/grey species)
        n_medium: Number of medium negatives (same species as positives, but normal)
        n_easy: Number of easy negatives (random from remaining)
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    negatives = all_observations_df[
        ~all_observations_df["observation_id"].isin(positive_obs_ids)
    ].copy()

    # Hard negatives: species with natural blue coloration
    hard_mask = negatives["species"].isin(HARD_NEGATIVE_SPECIES)
    hard_pool = negatives[hard_mask]
    hard = hard_pool.sample(n=min(n_hard, len(hard_pool)), random_state=rng)
    hard["negative_tier"] = "hard"

    # Medium negatives: same species as positives, but normal
    remaining = negatives[~negatives.index.isin(hard.index)]
    medium_mask = remaining["species"].isin(positive_species)
    medium_pool = remaining[medium_mask]
    medium = medium_pool.sample(n=min(n_medium, len(medium_pool)), random_state=rng)
    medium["negative_tier"] = "medium"

    # Easy negatives: stratified random from the rest
    remaining = remaining[~remaining.index.isin(medium.index)]
    easy = remaining.sample(n=min(n_easy, len(remaining)), random_state=rng)
    easy["negative_tier"] = "easy"

    result = pd.concat([hard, medium, easy], ignore_index=True)
    result["label"] = 0
    logger.info(
        f"Curated negatives: {len(hard)} hard, {len(medium)} medium, {len(easy)} easy"
    )
    return result
