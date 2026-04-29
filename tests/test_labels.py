"""Tests for Womack et al. label loading and parsing."""

import pandas as pd
from blue_frogs.data.labels import (
    parse_observation_id,
    load_womack_labels,
    filter_inat_records,
)


def test_parse_observation_id_standard_url():
    assert parse_observation_id("inaturalist.org/obs/12345678") == 12345678


def test_parse_observation_id_full_url():
    assert parse_observation_id("https://www.inaturalist.org/observations/12345678") == 12345678


def test_parse_observation_id_non_inat():
    assert parse_observation_id("Hinz, 1976") is None


def test_parse_observation_id_empty():
    assert parse_observation_id("") is None


def test_load_womack_labels(sample_womack_csv):
    df = load_womack_labels(sample_womack_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "observation_id" in df.columns


def test_filter_inat_records(sample_womack_csv):
    df = load_womack_labels(sample_womack_csv)
    inat_df = filter_inat_records(df)
    assert len(inat_df) == 3  # only iNat records
    assert all(inat_df["observation_id"].notna())
    assert set(inat_df["observation_id"]) == {12345678, 87654321, 11111111}
