"""Shared test fixtures."""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory with expected subdirectories."""
    for subdir in ["raw_images", "curated", "splits"]:
        (tmp_path / subdir).mkdir()
    return tmp_path


@pytest.fixture
def sample_womack_csv(tmp_path):
    """Minimal Womack-format CSV for testing."""
    data = pd.DataFrame({
        "Source": ["iNat", "iNat", "Sci Lit", "iNat"],
        "Family": ["Hylidae", "Ranidae", "Hylidae", "Hylidae"],
        "Genera": ["Hyla", "Lithobates", "Hyla", "Agalychnis"],
        "Species": [
            "Hyla cinerea",
            "Lithobates clamitans",
            "Hyla arborea",
            "Agalychnis callidryas",
        ],
        "Location": ["Virginia", "New York", "France", "Costa Rica"],
        "Latitude": [37.5, 42.3, 48.8, 10.4],
        "Longitude": [-77.4, -73.9, 2.3, -84.1],
        "Unusual Pattern": ["Blue", "Blue", "Blue", "dorsal axanthic"],
        "Lifestage": ["Adult", "Adult", "Adult", "Adult"],
        "Citation": [
            "inaturalist.org/obs/12345678",
            "inaturalist.org/obs/87654321",
            "Hinz, 1976",
            "inaturalist.org/obs/11111111",
        ],
        "Year": [2020, 2019, 1976, 2021],
    })
    csv_path = tmp_path / "axanthic_records.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_image(tmp_path):
    """Create a small test image."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(0, 128, 0))  # green frog-ish
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


@pytest.fixture
def sample_blue_image(tmp_path):
    """Create a small blue test image (simulating axanthism)."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(50, 100, 200))  # blue-ish
    path = tmp_path / "test_blue_image.jpg"
    img.save(path)
    return path
