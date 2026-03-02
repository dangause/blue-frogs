"""Tests for training data preparation script."""

import pytest
from PIL import Image


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a minimal mock dataset with images and a Womack-style CSV."""
    image_dir = tmp_path / "raw_images"

    # Create 5 positive observations (matching Womack CSV format)
    positive_obs = [100456051, 75584907, 19547423, 12345678, 87654321]
    for obs_id in positive_obs:
        obs_dir = image_dir / str(obs_id)
        obs_dir.mkdir(parents=True)
        # 1-2 photos per observation
        for photo_idx in range(1, 3):
            photo_id = obs_id * 10 + photo_idx
            img = Image.new("RGB", (100, 100), color=(50, 100, 200))
            img.save(obs_dir / f"{photo_id}.jpg")

    # Create 20 negative observations (not in Womack CSV)
    for i in range(20):
        obs_id = 9000000 + i
        obs_dir = image_dir / str(obs_id)
        obs_dir.mkdir(parents=True)
        photo_id = obs_id * 10 + 1
        img = Image.new("RGB", (100, 100), color=(0, 128, 0))
        img.save(obs_dir / f"{photo_id}.jpg")

    # Create one empty observation dir (should be skipped)
    (image_dir / "9999999").mkdir(parents=True)

    # Create Womack-style CSV
    csv_path = tmp_path / "womack_labels.csv"
    lines = [
        "Source,Family,Genera,Species,Individuals Observed,Location,Longitude,Latitude,Country,Usual Pattern,Unusual Pattern,Pattern Notes,Iris color,Lifestage,Citation,Year,Notes",
    ]
    for obs_id in positive_obs:
        lines.append(
            f'iNat,Hylidae,Agalychnis,Agalychnis callidryas,1,Costa Rica,10.0,-85.0,Costa Rica,Green,Blue,Whole body,N/A,Adult,https://www.inaturalist.org/observations/{obs_id},2021,'
        )
    # Add one non-iNat record (should be ignored)
    lines.append(
        'Sci Lit,Hylidae,Acris,Acris crepitans,1,Virginia,-78.0,37.0,USA,,dorsal axanthic,,NA,juvenile,Volume 44 Issue,,',
    )
    csv_path.write_text("\n".join(lines))

    return tmp_path, csv_path


def test_build_labels_json(mock_dataset):
    """build_labels produces correct label assignments."""
    from scripts.prepare_training_data import build_labels

    data_dir, csv_path = mock_dataset
    labels = build_labels(data_dir / "raw_images", csv_path)

    # 5 positive obs * 2 photos + 20 negative obs * 1 photo = 30
    assert len(labels) == 30

    positives = [e for e in labels if e["label"] == 1]
    negatives = [e for e in labels if e["label"] == 0]
    assert len(positives) == 10  # 5 obs * 2 photos
    assert len(negatives) == 20

    # Check structure
    for entry in labels:
        assert "observation_id" in entry
        assert "photo_id" in entry
        assert "photo_path" in entry
        assert "label" in entry
        assert "/" in entry["photo_path"]  # obs_id/photo_id.jpg format


def test_build_splits_by_observation(mock_dataset):
    """build_splits splits by observation, not photo, preventing leakage."""
    from scripts.prepare_training_data import build_labels, build_splits

    data_dir, csv_path = mock_dataset
    labels = build_labels(data_dir / "raw_images", csv_path)
    splits = build_splits(labels, test_fraction=0.2, seed=42)

    train_indices = splits["train_indices"]
    test_indices = splits["test_indices"]

    assert len(train_indices) + len(test_indices) == len(labels)
    assert set(train_indices) & set(test_indices) == set()

    # No observation appears in both train and test
    train_obs = {labels[i]["observation_id"] for i in train_indices}
    test_obs = {labels[i]["observation_id"] for i in test_indices}
    assert train_obs & test_obs == set()
