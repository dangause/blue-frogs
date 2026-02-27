"""Dataset splitting with stratification."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def create_stratified_splits(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and held-out test, stratified by label.

    Splits by observation (not photo) to prevent data leakage.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_fold_indices(
    train_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified k-fold indices for cross-validation.

    Returns list of (train_indices, val_indices) tuples.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(train_df, train_df["label"]):
        folds.append((train_idx, val_idx))
    return folds
