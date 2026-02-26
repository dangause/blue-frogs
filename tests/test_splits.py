"""Tests for dataset curation and splitting."""

import pytest
import pandas as pd
import numpy as np
from blue_frogs.data.splits import (
    create_stratified_splits,
    get_fold_indices,
)


@pytest.fixture
def curated_dataset():
    """Simulated curated dataset with positives and tiered negatives."""
    rng = np.random.RandomState(42)
    n_pos, n_neg = 50, 500
    df = pd.DataFrame({
        "observation_id": range(n_pos + n_neg),
        "label": [1] * n_pos + [0] * n_neg,
        "genus": (
            rng.choice(["Hyla", "Lithobates", "Agalychnis"], n_pos).tolist()
            + rng.choice(["Hyla", "Lithobates", "Rana", "Bufo", "Dendrobates"], n_neg).tolist()
        ),
        "negative_tier": (
            [None] * n_pos
            + rng.choice(["hard", "medium", "easy"], n_neg).tolist()
        ),
    })
    return df


def test_create_stratified_splits_test_fraction(curated_dataset):
    train_df, test_df = create_stratified_splits(
        curated_dataset, test_fraction=0.2, seed=42
    )
    # Test set should have ~20% of positives
    test_positives = test_df["label"].sum()
    total_positives = curated_dataset["label"].sum()
    assert 0.15 <= test_positives / total_positives <= 0.25


def test_create_stratified_splits_no_observation_leak(curated_dataset):
    train_df, test_df = create_stratified_splits(
        curated_dataset, test_fraction=0.2, seed=42
    )
    train_ids = set(train_df["observation_id"])
    test_ids = set(test_df["observation_id"])
    assert train_ids.isdisjoint(test_ids)


def test_get_fold_indices_correct_count(curated_dataset):
    train_df, _ = create_stratified_splits(curated_dataset, test_fraction=0.2, seed=42)
    folds = get_fold_indices(train_df, n_folds=5, seed=42)
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert len(set(train_idx) & set(val_idx)) == 0


def test_get_fold_indices_preserves_positives(curated_dataset):
    train_df, _ = create_stratified_splits(curated_dataset, test_fraction=0.2, seed=42)
    folds = get_fold_indices(train_df, n_folds=5, seed=42)
    for train_idx, val_idx in folds:
        val_labels = train_df.iloc[val_idx]["label"]
        assert val_labels.sum() > 0, "Each fold must contain positive examples"
