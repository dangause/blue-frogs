"""Tests for batch inference pipeline."""

import pytest
import pandas as pd
import numpy as np
from blue_frogs.inference.batch_scorer import (
    format_prediction_row,
    filter_flagged_predictions,
)


def test_format_prediction_row():
    row = format_prediction_row(
        observation_id=12345,
        photo_id=111,
        score=0.87,
        threshold=0.5,
        model_version="model_a_v1",
    )
    assert row["observation_id"] == 12345
    assert row["predicted_class"] == 1
    assert row["prediction_score"] == 0.87


def test_filter_flagged_predictions():
    df = pd.DataFrame({
        "observation_id": [1, 2, 3, 4],
        "prediction_score": [0.1, 0.6, 0.9, 0.3],
    })
    flagged = filter_flagged_predictions(df, threshold=0.5)
    assert len(flagged) == 2
    assert set(flagged["observation_id"]) == {2, 3}
