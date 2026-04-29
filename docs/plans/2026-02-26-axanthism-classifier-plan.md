# Axanthism Classifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CV pipeline to detect axanthism in frogs from iNaturalist images, comparing three model architectures, and batch-score the full iNat frog corpus.

**Architecture:** PyTorch Lightning training pipeline with shared data/eval infrastructure across three model families (EfficientNetV2-S ensemble, two-stage detection+classification, DINOv2/BioCLIP foundation models). Data from Womack et al. 2025 Blue Frogs Project via iNaturalist API.

**Tech Stack:** Python 3.11, PyTorch 2.x, PyTorch Lightning, timm, albumentations, pyinaturalist, Weights & Biases, scipy, statsmodels, matplotlib, seaborn

---

## Project Structure

```
blue-frogs/
  pyproject.toml
  environment.yml
  src/
    blue_frogs/
      __init__.py
      config.py
      data/
        __init__.py
        inat_client.py
        downloader.py
        labels.py
        curation.py
        splits.py
        dataset.py
        color_features.py
      models/
        __init__.py
        common.py
        model_a.py
        model_b_detector.py
        model_b_classifier.py
        model_c.py
      evaluation/
        __init__.py
        metrics.py
        comparison.py
        interpretability.py
      inference/
        __init__.py
        batch_scorer.py
      figures/
        __init__.py
        paper_figures.py
  tests/
    conftest.py
    test_labels.py
    test_splits.py
    test_dataset.py
    test_color_features.py
    test_model_a.py
    test_model_b.py
    test_model_c.py
    test_metrics.py
    test_batch_scorer.py
  scripts/
    download_images.py
    curate_training_set.py
    train.py
    compare_models.py
    run_inference.py
    generate_figures.py
  configs/
    model_a.yaml
    model_b.yaml
    model_c.yaml
  docs/
    plans/
```

---

## Task 1: Project Scaffold & Environment

**Files:**
- Create: `pyproject.toml`
- Create: `environment.yml`
- Create: `src/blue_frogs/__init__.py`
- Create: `src/blue_frogs/config.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "blue-frogs"
version = "0.1.0"
description = "CV pipeline for detecting axanthism in frogs from iNaturalist images"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "torchvision>=0.16",
    "pytorch-lightning>=2.1",
    "timm>=0.9",
    "albumentations>=1.3",
    "pyinaturalist>=0.19",
    "wandb>=0.16",
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "statsmodels>=0.14",
    "scikit-learn>=1.3",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pillow>=10.0",
    "opencv-python-headless>=4.8",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "requests>=2.31",
    "aiohttp>=3.9",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.1"]
bioclip = ["open_clip_torch>=2.23"]
yolo = ["ultralytics>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py311"
```

**Step 2: Create environment.yml**

```yaml
name: blue-frogs
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch>=2.1
  - torchvision>=0.16
  - pytorch-cuda=12.1
  - pip
  - pip:
    - -e ".[dev,bioclip,yolo]"
```

**Step 3: Create src/blue_frogs/__init__.py**

```python
"""Blue Frogs: CV pipeline for detecting axanthism in frogs."""
```

**Step 4: Create src/blue_frogs/config.py**

```python
"""Central configuration for the Blue Frogs pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGE_DIR = DATA_DIR / "raw_images"
CURATED_DIR = DATA_DIR / "curated"
SPLITS_DIR = DATA_DIR / "splits"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# iNaturalist
ANURA_TAXON_ID = 20979
INAT_API_BASE = "https://api.inaturalist.org/v1"
INAT_RATE_LIMIT = 60  # requests per minute
INAT_PHOTO_BASE = "https://static.inaturalist.org/photos"

# Reproducibility
RANDOM_SEED = 42
N_FOLDS = 5
TEST_FRACTION = 0.2

# Training defaults
IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Womack et al. data
WOMACK_GITHUB_URL = (
    "https://raw.githubusercontent.com/mcwomack/bluefrogs/main/AxanthicRecords_Upload.csv"
)
```

**Step 5: Create tests/conftest.py**

```python
"""Shared test fixtures."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile


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
        "Species": ["Hyla cinerea", "Lithobates clamitans", "Hyla arborea", "Agalychnis callidryas"],
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
```

**Step 6: Create directory structure**

```bash
mkdir -p src/blue_frogs/{data,models,evaluation,inference,figures}
mkdir -p tests scripts configs
touch src/blue_frogs/data/__init__.py
touch src/blue_frogs/models/__init__.py
touch src/blue_frogs/evaluation/__init__.py
touch src/blue_frogs/inference/__init__.py
touch src/blue_frogs/figures/__init__.py
```

**Step 7: Update .gitignore with project-specific entries**

Append to existing `.gitignore`:
```
# Project data (large files, not in git)
data/
models/
results/
figures/
wandb/
*.ckpt
*.pth
```

**Step 8: Install and verify**

```bash
conda env create -f environment.yml
conda activate blue-frogs
pip install -e ".[dev]"
pytest tests/ -v
```

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with environment and config"
```

---

## Task 2: Label Loading & Parsing

**Files:**
- Create: `src/blue_frogs/data/labels.py`
- Create: `tests/test_labels.py`

**Step 1: Write the failing tests**

```python
# tests/test_labels.py
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_labels.py -v
```
Expected: FAIL (module not found)

**Step 3: Implement labels.py**

```python
# src/blue_frogs/data/labels.py
"""Load and parse labeled data from the Womack et al. Blue Frogs Project."""

import re
from pathlib import Path

import pandas as pd
import requests

from blue_frogs.config import WOMACK_GITHUB_URL

# Patterns to extract iNaturalist observation IDs from citation strings
_INAT_PATTERNS = [
    re.compile(r"inaturalist\.org/obs(?:ervations)?/(\d+)"),
    re.compile(r"inaturalist\.org/\S*?(\d{6,})"),
]


def parse_observation_id(citation: str) -> int | None:
    """Extract iNaturalist observation ID from a citation string.

    Returns None if the citation is not an iNaturalist URL.
    """
    if not citation or not isinstance(citation, str):
        return None
    for pattern in _INAT_PATTERNS:
        match = pattern.search(citation)
        if match:
            return int(match.group(1))
    return None


def load_womack_labels(path: str | Path) -> pd.DataFrame:
    """Load the Womack et al. AxanthicRecords CSV and add parsed observation IDs."""
    df = pd.read_csv(path)
    df["observation_id"] = df["Citation"].apply(parse_observation_id)
    return df


def filter_inat_records(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only iNaturalist records with valid observation IDs."""
    mask = (df["Source"] == "iNat") & df["observation_id"].notna()
    result = df[mask].copy()
    result["observation_id"] = result["observation_id"].astype(int)
    return result


def download_womack_labels(save_path: str | Path) -> Path:
    """Download the Womack et al. CSV from GitHub."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(WOMACK_GITHUB_URL, timeout=30)
    response.raise_for_status()
    save_path.write_text(response.text)
    return save_path
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_labels.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/blue_frogs/data/labels.py tests/test_labels.py
git commit -m "feat: label loading and iNat observation ID parsing"
```

---

## Task 3: iNaturalist API Client

**Files:**
- Create: `src/blue_frogs/data/inat_client.py`
- Create: `tests/test_inat_client.py`

**Step 1: Write the failing tests**

```python
# tests/test_inat_client.py
"""Tests for iNaturalist API client."""

import pytest
from unittest.mock import patch, MagicMock
from blue_frogs.data.inat_client import (
    get_photo_urls,
    build_medium_url,
    fetch_observation_metadata,
)


def test_build_medium_url_from_square():
    square_url = "https://static.inaturalist.org/photos/61482854/square.jpg?1581761020"
    result = build_medium_url(square_url)
    assert "medium" in result
    assert "square" not in result


def test_build_medium_url_from_other_size():
    url = "https://static.inaturalist.org/photos/61482854/small.jpg"
    result = build_medium_url(url)
    assert "/medium." in result


def test_get_photo_urls_extracts_all_photos():
    observation = {
        "id": 12345,
        "observation_photos": [
            {"photo": {"id": 111, "url": "https://static.inaturalist.org/photos/111/square.jpg"}},
            {"photo": {"id": 222, "url": "https://static.inaturalist.org/photos/222/square.jpg"}},
        ],
    }
    urls = get_photo_urls(observation)
    assert len(urls) == 2
    assert all("medium" in u for u in urls)


def test_get_photo_urls_empty_observation():
    observation = {"id": 12345, "observation_photos": []}
    assert get_photo_urls(observation) == []


@patch("blue_frogs.data.inat_client.get_observations")
def test_fetch_observation_metadata(mock_get):
    mock_get.return_value = {
        "results": [
            {
                "id": 12345,
                "quality_grade": "research",
                "taxon": {"id": 1234, "name": "Hyla cinerea"},
                "observed_on": "2021-06-15",
                "location": "37.5,-77.4",
                "observation_photos": [
                    {"photo": {"id": 111, "url": "https://example.com/photos/111/square.jpg"}}
                ],
            }
        ]
    }
    result = fetch_observation_metadata([12345])
    assert len(result) == 1
    assert result[0]["id"] == 12345
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inat_client.py -v
```

**Step 3: Implement inat_client.py**

```python
# src/blue_frogs/data/inat_client.py
"""iNaturalist API client for fetching observation metadata and photo URLs."""

import re
import time
from typing import Any

from pyinaturalist import get_observations

from blue_frogs.config import ANURA_TAXON_ID, INAT_RATE_LIMIT

_SIZE_PATTERN = re.compile(r"/(square|small|thumb|large|original)\.")


def build_medium_url(photo_url: str) -> str:
    """Convert any iNaturalist photo URL to medium size (1024px longest side)."""
    return _SIZE_PATTERN.sub("/medium.", photo_url)


def get_photo_urls(observation: dict) -> list[str]:
    """Extract medium-resolution photo URLs from an observation dict."""
    urls = []
    for obs_photo in observation.get("observation_photos", []):
        photo = obs_photo.get("photo", {})
        url = photo.get("url")
        if url:
            urls.append(build_medium_url(url))
    return urls


def fetch_observation_metadata(
    observation_ids: list[int],
    batch_size: int = 30,
) -> list[dict[str, Any]]:
    """Fetch metadata for a list of observation IDs from the iNaturalist API.

    Batches requests to respect rate limits.
    """
    results = []
    for i in range(0, len(observation_ids), batch_size):
        batch = observation_ids[i : i + batch_size]
        response = get_observations(id=batch, per_page=batch_size)
        results.extend(response.get("results", []))
        # Rate limiting
        if i + batch_size < len(observation_ids):
            time.sleep(60 / INAT_RATE_LIMIT)
    return results


def fetch_anura_observations_page(
    id_above: int = 0,
    per_page: int = 200,
) -> dict[str, Any]:
    """Fetch a single page of research-grade Anura observations.

    Use id_above for cursor-based pagination through the full corpus.
    """
    return get_observations(
        taxon_id=ANURA_TAXON_ID,
        quality_grade="research",
        photos=True,
        per_page=per_page,
        id_above=id_above,
        order_by="id",
        order="asc",
    )
```

**Step 4: Run tests**

```bash
pytest tests/test_inat_client.py -v
```

**Step 5: Commit**

```bash
git add src/blue_frogs/data/inat_client.py tests/test_inat_client.py
git commit -m "feat: iNaturalist API client with photo URL extraction"
```

---

## Task 4: Image Downloader

**Files:**
- Create: `src/blue_frogs/data/downloader.py`
- Create: `scripts/download_images.py`

**Step 1: Write the failing test**

```python
# tests/test_downloader.py
"""Tests for image downloader."""

import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path
from blue_frogs.data.downloader import (
    build_download_manifest,
    download_image_sync,
)


def test_build_download_manifest():
    observations = [
        {
            "id": 123,
            "observation_photos": [
                {"photo": {"id": 111, "url": "https://example.com/photos/111/square.jpg"}},
                {"photo": {"id": 222, "url": "https://example.com/photos/222/square.jpg"}},
            ],
        },
    ]
    manifest = build_download_manifest(observations, Path("/tmp/images"))
    assert len(manifest) == 2
    assert manifest[0]["photo_id"] == 111
    assert manifest[0]["save_path"] == Path("/tmp/images/123/111.jpg")
    assert "medium" in manifest[0]["url"]


@patch("blue_frogs.data.downloader.requests.get")
def test_download_image_sync(mock_get, tmp_path):
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.content = b"fake image data"
    mock_response.raise_for_status = lambda: None

    save_path = tmp_path / "test.jpg"
    result = download_image_sync("https://example.com/photo.jpg", save_path)
    assert result is True
    assert save_path.read_bytes() == b"fake image data"
```

**Step 2: Implement downloader.py**

```python
# src/blue_frogs/data/downloader.py
"""Download images from iNaturalist."""

import logging
import time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from blue_frogs.data.inat_client import build_medium_url

logger = logging.getLogger(__name__)


def build_download_manifest(
    observations: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Build a list of (url, save_path) entries from observation metadata."""
    manifest = []
    for obs in observations:
        obs_id = obs["id"]
        obs_dir = output_dir / str(obs_id)
        for obs_photo in obs.get("observation_photos", []):
            photo = obs_photo.get("photo", {})
            photo_id = photo.get("id")
            url = photo.get("url")
            if not url or not photo_id:
                continue
            medium_url = build_medium_url(url)
            ext = "jpg"
            save_path = obs_dir / f"{photo_id}.{ext}"
            manifest.append({
                "observation_id": obs_id,
                "photo_id": photo_id,
                "url": medium_url,
                "save_path": save_path,
            })
    return manifest


def download_image_sync(url: str, save_path: Path, timeout: int = 30) -> bool:
    """Download a single image. Returns True on success."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            return True  # skip already downloaded
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        save_path.write_bytes(response.content)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def download_batch(
    manifest: list[dict[str, Any]],
    max_workers: int = 8,
    rate_limit: float = 60,
) -> dict[str, int]:
    """Download images in parallel with rate limiting.

    Returns dict with counts: {"success": N, "failed": N, "skipped": N}.
    """
    stats = {"success": 0, "failed": 0, "skipped": 0}
    delay = 1.0 / rate_limit if rate_limit > 0 else 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for entry in manifest:
            if entry["save_path"].exists():
                stats["skipped"] += 1
                continue
            future = executor.submit(
                download_image_sync, entry["url"], entry["save_path"]
            )
            futures[future] = entry
            time.sleep(delay)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if future.result():
                stats["success"] += 1
            else:
                stats["failed"] += 1

    return stats
```

**Step 3: Create download script**

```python
# scripts/download_images.py
"""CLI script to download images for the training set."""

import argparse
import logging
from pathlib import Path

from blue_frogs.config import RAW_IMAGE_DIR, DATA_DIR
from blue_frogs.data.labels import load_womack_labels, filter_inat_records, download_womack_labels
from blue_frogs.data.inat_client import fetch_observation_metadata
from blue_frogs.data.downloader import build_download_manifest, download_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download frog images from iNaturalist")
    parser.add_argument("--labels-csv", type=Path, default=None,
                        help="Path to Womack labels CSV. Downloads from GitHub if not provided.")
    parser.add_argument("--output-dir", type=Path, default=RAW_IMAGE_DIR)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--positives-only", action="store_true",
                        help="Only download confirmed axanthic observations")
    args = parser.parse_args()

    # Load labels
    if args.labels_csv is None:
        csv_path = DATA_DIR / "womack_labels.csv"
        if not csv_path.exists():
            logger.info("Downloading Womack labels from GitHub...")
            download_womack_labels(csv_path)
        args.labels_csv = csv_path

    df = load_womack_labels(args.labels_csv)
    inat_df = filter_inat_records(df)
    obs_ids = inat_df["observation_id"].tolist()
    logger.info(f"Found {len(obs_ids)} iNaturalist observation IDs")

    # Fetch metadata from API
    logger.info("Fetching observation metadata from iNaturalist API...")
    metadata = fetch_observation_metadata(obs_ids)
    logger.info(f"Retrieved metadata for {len(metadata)} observations")

    # Build manifest and download
    manifest = build_download_manifest(metadata, args.output_dir)
    logger.info(f"Download manifest: {len(manifest)} images")

    stats = download_batch(manifest, max_workers=args.max_workers)
    logger.info(f"Download complete: {stats}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests and commit**

```bash
pytest tests/test_downloader.py -v
git add src/blue_frogs/data/downloader.py tests/test_downloader.py scripts/download_images.py
git commit -m "feat: parallel image downloader with rate limiting"
```

---

## Task 5: Training Set Curation & Splits

**Files:**
- Create: `src/blue_frogs/data/curation.py`
- Create: `src/blue_frogs/data/splits.py`
- Create: `tests/test_splits.py`

**Step 1: Write failing tests**

```python
# tests/test_splits.py
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
```

**Step 2: Implement curation.py**

```python
# src/blue_frogs/data/curation.py
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
```

**Step 3: Implement splits.py**

```python
# src/blue_frogs/data/splits.py
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
```

**Step 4: Run tests and commit**

```bash
pytest tests/test_splits.py -v
git add src/blue_frogs/data/curation.py src/blue_frogs/data/splits.py tests/test_splits.py
git commit -m "feat: training set curation with tiered negatives and stratified splits"
```

---

## Task 6: PyTorch Dataset & Augmentation Pipeline

**Files:**
- Create: `src/blue_frogs/data/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write failing tests**

```python
# tests/test_dataset.py
"""Tests for PyTorch dataset and augmentation."""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from blue_frogs.data.dataset import (
    FrogDataset,
    get_train_transforms,
    get_val_transforms,
)


@pytest.fixture
def image_dir(tmp_path):
    """Create fake image directory structure."""
    for obs_id, color in [(123, (0, 128, 0)), (456, (50, 100, 200))]:
        obs_dir = tmp_path / str(obs_id)
        obs_dir.mkdir()
        img = Image.new("RGB", (200, 150), color=color)
        img.save(obs_dir / "001.jpg")
    return tmp_path


@pytest.fixture
def sample_metadata():
    return [
        {"observation_id": 123, "photo_id": 1, "photo_path": "123/001.jpg", "label": 0},
        {"observation_id": 456, "photo_id": 2, "photo_path": "456/001.jpg", "label": 1},
    ]


def test_frog_dataset_len(image_dir, sample_metadata):
    ds = FrogDataset(sample_metadata, image_dir, transform=get_val_transforms())
    assert len(ds) == 2


def test_frog_dataset_getitem_shape(image_dir, sample_metadata):
    ds = FrogDataset(sample_metadata, image_dir, transform=get_val_transforms())
    img, label = ds[0]
    assert img.shape == (3, 384, 384)
    assert isinstance(label, (int, float))


def test_train_transforms_preserve_shape(sample_image):
    transforms = get_train_transforms()
    img = np.array(Image.open(sample_image))
    result = transforms(image=img)
    assert result["image"].shape == (3, 384, 384)


def test_val_transforms_deterministic(sample_image):
    transforms = get_val_transforms()
    img = np.array(Image.open(sample_image))
    r1 = transforms(image=img)["image"]
    r2 = transforms(image=img)["image"]
    assert torch.allclose(r1, r2)
```

**Step 2: Implement dataset.py**

```python
# src/blue_frogs/data/dataset.py
"""PyTorch Dataset and augmentation pipeline for frog images."""

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from blue_frogs.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> A.Compose:
    """Training augmentation pipeline.

    Hue augmentation is capped at +/- 5 degrees (out of 180 in OpenCV)
    because axanthism is a color-based trait.
    """
    return A.Compose([
        A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.014,  # ~5 degrees / 360 = 0.014
            p=0.5,
        ),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Validation/test transforms (deterministic)."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class FrogDataset(Dataset):
    """Dataset for frog images with binary axanthism labels."""

    def __init__(
        self,
        metadata: list[dict[str, Any]],
        image_dir: Path,
        transform: A.Compose | None = None,
    ):
        self.metadata = metadata
        self.image_dir = Path(image_dir)
        self.transform = transform or get_val_transforms()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.metadata[idx]
        img_path = self.image_dir / entry["photo_path"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        label = entry["label"]

        return image_tensor, label
```

**Step 3: Run tests and commit**

```bash
pytest tests/test_dataset.py -v
git add src/blue_frogs/data/dataset.py tests/test_dataset.py
git commit -m "feat: PyTorch dataset with hue-constrained augmentation pipeline"
```

---

## Task 7: Shared Training Infrastructure

**Files:**
- Create: `src/blue_frogs/models/common.py`
- Create: `src/blue_frogs/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_metrics.py
"""Tests for evaluation metrics."""

import pytest
import numpy as np
from blue_frogs.evaluation.metrics import (
    compute_classification_metrics,
    compute_bootstrap_ci,
    find_threshold_at_recall,
    find_threshold_at_precision,
)


def test_compute_classification_metrics_perfect():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    metrics = compute_classification_metrics(y_true, y_score)
    assert metrics["auroc"] > 0.99
    assert metrics["auprc"] > 0.99


def test_compute_classification_metrics_random():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 100)
    y_score = rng.random(100)
    metrics = compute_classification_metrics(y_true, y_score)
    assert 0.0 < metrics["auroc"] < 1.0
    assert "f1_optimal" in metrics
    assert "optimal_threshold" in metrics


def test_compute_bootstrap_ci():
    rng = np.random.RandomState(42)
    y_true = np.array([0] * 90 + [1] * 10)
    y_score = np.concatenate([rng.random(90) * 0.5, 0.5 + rng.random(10) * 0.5])
    ci = compute_bootstrap_ci(y_true, y_score, metric="auroc", n_bootstrap=100, seed=42)
    assert ci["lower"] < ci["mean"] < ci["upper"]


def test_find_threshold_at_recall():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    threshold = find_threshold_at_recall(y_true, y_score, target_recall=0.95)
    assert 0.0 <= threshold <= 1.0


def test_find_threshold_at_precision():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    threshold = find_threshold_at_precision(y_true, y_score, target_precision=0.95)
    assert 0.0 <= threshold <= 1.0
```

**Step 2: Implement metrics.py**

```python
# src/blue_frogs/evaluation/metrics.py
"""Evaluation metrics for binary classification with class imbalance."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Compute all classification metrics for a single evaluation."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # F1 at each threshold
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_score >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1_optimal": float(f1_scores[best_idx]),
        "optimal_threshold": float(optimal_threshold),
        "precision_at_optimal": float(precision[best_idx]),
        "recall_at_optimal": float(recall[best_idx]),
        "confusion_matrix": cm.tolist(),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auroc",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence intervals for a metric."""
    rng = np.random.RandomState(seed)
    metric_fn = {"auroc": roc_auc_score, "auprc": average_precision_score}[metric]

    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))

    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(scores)),
        "lower": float(np.percentile(scores, 100 * alpha)),
        "upper": float(np.percentile(scores, 100 * (1 - alpha))),
    }


def find_threshold_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.95
) -> float:
    """Find the decision threshold that achieves at least target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = recall >= target_recall
    if not valid.any():
        return 0.0
    # Among thresholds achieving target recall, pick highest (most conservative)
    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(precision[valid_idx])]
    return float(thresholds[best]) if best < len(thresholds) else 0.0


def find_threshold_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.95
) -> float:
    """Find the decision threshold that achieves at least target_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = precision >= target_precision
    if not valid.any():
        return 1.0
    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(recall[valid_idx])]
    return float(thresholds[best]) if best < len(thresholds) else 1.0
```

**Step 3: Implement common.py (shared model components)**

```python
# src/blue_frogs/models/common.py
"""Shared model components: losses, base Lightning module, samplers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler

from blue_frogs.evaluation.metrics import compute_classification_metrics


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class BaseClassifier(pl.LightningModule):
    """Base Lightning module for all frog classifiers."""

    def __init__(
        self,
        loss_type: str = "weighted_bce",
        pos_weight: float = 20.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "sgd",
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        if loss_type == "weighted_bce":
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Collect predictions for epoch-level metrics
        self._val_outputs: list[dict] = []

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits.squeeze(-1), labels.float())

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits.squeeze(-1))
        self._val_outputs.append({
            "loss": loss,
            "probs": probs.detach().cpu(),
            "labels": labels.detach().cpu(),
        })
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat([o["probs"] for o in self._val_outputs]).numpy()
        all_labels = torch.cat([o["labels"] for o in self._val_outputs]).numpy()
        avg_loss = torch.stack([o["loss"] for o in self._val_outputs]).mean()

        if len(set(all_labels)) >= 2:
            metrics = compute_classification_metrics(all_labels, all_probs)
            self.log("val/loss", avg_loss, prog_bar=True)
            self.log("val/auprc", metrics["auprc"], prog_bar=True)
            self.log("val/auroc", metrics["auroc"])
            self.log("val/f1", metrics["f1_optimal"])

        self._val_outputs.clear()

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create a weighted random sampler that oversamples the minority class."""
    import numpy as np
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    weights = 1.0 / class_counts[labels_arr]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(labels),
        replacement=True,
    )
```

**Step 4: Run tests and commit**

```bash
pytest tests/test_metrics.py -v
git add src/blue_frogs/models/common.py src/blue_frogs/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: shared training infra - focal loss, base classifier, metrics"
```

---

## Task 8: Model A - EfficientNetV2-S Ensemble

**Files:**
- Create: `src/blue_frogs/models/model_a.py`
- Create: `configs/model_a.yaml`
- Create: `scripts/train.py`
- Create: `tests/test_model_a.py`

**Step 1: Write failing tests**

```python
# tests/test_model_a.py
"""Tests for Model A: EfficientNetV2-S classifier."""

import pytest
import torch
from blue_frogs.models.model_a import EfficientNetClassifier


def test_efficientnet_forward_shape():
    model = EfficientNetClassifier(backbone="tf_efficientnetv2_s")
    x = torch.randn(2, 3, 384, 384)
    logits = model(x)
    assert logits.shape == (2, 1)


def test_efficientnet_training_step():
    model = EfficientNetClassifier(backbone="tf_efficientnetv2_s")
    x = torch.randn(2, 3, 384, 384)
    labels = torch.tensor([0, 1])
    loss = model.training_step((x, labels), 0)
    assert loss.item() > 0


def test_efficientnet_frozen_backbone():
    model = EfficientNetClassifier(backbone="tf_efficientnetv2_s", freeze_backbone=True)
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen"
```

**Step 2: Implement model_a.py**

```python
# src/blue_frogs/models/model_a.py
"""Model A: EfficientNetV2-S ensemble classifier."""

import timm
import torch
import torch.nn as nn

from blue_frogs.models.common import BaseClassifier


class EfficientNetClassifier(BaseClassifier):
    """EfficientNetV2-S with binary classification head."""

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class EfficientNetEnsemble:
    """Ensemble of EfficientNetV2-S models from k-fold training."""

    def __init__(self, checkpoint_paths: list[str]):
        self.models = []
        for path in checkpoint_paths:
            model = EfficientNetClassifier.load_from_checkpoint(path)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all ensemble members."""
        logits = torch.stack([m(x) for m in self.models])
        avg_logits = logits.mean(dim=0)
        return torch.sigmoid(avg_logits.squeeze(-1))
```

**Step 3: Create config file**

```yaml
# configs/model_a.yaml
model:
  backbone: tf_efficientnetv2_s
  pretrained: true
  dropout: 0.3
  freeze_backbone: false

training:
  loss_type: weighted_bce
  pos_weight: 20.0
  optimizer: sgd
  learning_rate: 0.001
  weight_decay: 0.0
  max_epochs: 80
  batch_size: 32
  early_stopping_patience: 10
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

data:
  image_size: 384
  n_folds: 5
  seed: 42

ablation:
  # Also train with focal loss for comparison
  focal_loss:
    loss_type: focal
    focal_gamma: 2.0
```

**Step 4: Create training script (shared across all models)**

```python
# scripts/train.py
"""Unified training script for all model architectures."""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from blue_frogs.config import MODEL_DIR, RANDOM_SEED
from blue_frogs.data.dataset import FrogDataset, get_train_transforms, get_val_transforms
from blue_frogs.data.splits import get_fold_indices
from blue_frogs.models.common import make_weighted_sampler
from blue_frogs.models.model_a import EfficientNetClassifier
from blue_frogs.models.model_c import FoundationModelClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "model_a": EfficientNetClassifier,
    "model_c": FoundationModelClassifier,
}


def train_fold(
    model_class,
    model_kwargs: dict,
    train_dataset: FrogDataset,
    val_dataset: FrogDataset,
    config: dict,
    fold: int,
    output_dir: Path,
):
    """Train a single fold."""
    pl.seed_everything(RANDOM_SEED + fold)

    model = model_class(**model_kwargs)

    train_labels = [m["label"] for m in train_dataset.metadata]
    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir / f"fold_{fold}",
        filename="best-{val/auprc:.4f}",
        monitor="val/auprc",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/auprc",
        mode="max",
        patience=config["training"]["early_stopping_patience"],
    )
    wandb_logger = WandbLogger(
        project="blue-frogs",
        name=f"{config.get('model_name', 'model')}_fold{fold}",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=wandb_logger,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    return checkpoint_cb.best_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--fold", type=int, default=None, help="Train single fold (for HPC)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model_name"] = args.model
    model_class = MODEL_CLASSES[args.model]
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and splits -- details depend on splits file format
    # (This is filled in during execution based on actual data)
    logger.info(f"Training {args.model} with config {args.config}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
```

**Step 5: Run tests and commit**

```bash
pytest tests/test_model_a.py -v
git add src/blue_frogs/models/model_a.py configs/model_a.yaml scripts/train.py tests/test_model_a.py
git commit -m "feat: Model A - EfficientNetV2-S classifier with ensemble support"
```

---

## Task 9: Model B - Two-Stage Detection + Classification

**Files:**
- Create: `src/blue_frogs/data/color_features.py`
- Create: `src/blue_frogs/models/model_b_detector.py`
- Create: `src/blue_frogs/models/model_b_classifier.py`
- Create: `configs/model_b.yaml`
- Create: `tests/test_color_features.py`
- Create: `tests/test_model_b.py`

**Step 1: Write failing tests for color features**

```python
# tests/test_color_features.py
"""Tests for LAB color feature extraction."""

import numpy as np
import pytest
from PIL import Image
from blue_frogs.data.color_features import extract_lab_features


def test_extract_lab_features_shape(sample_image):
    img = np.array(Image.open(sample_image))
    features = extract_lab_features(img)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    assert len(features) == 30  # 10 features per channel (L, a, b)


def test_extract_lab_features_blue_vs_green(sample_image, sample_blue_image):
    green_img = np.array(Image.open(sample_image))
    blue_img = np.array(Image.open(sample_blue_image))
    green_feats = extract_lab_features(green_img)
    blue_feats = extract_lab_features(blue_img)
    # b* channel (indices 20-29) should be more negative for blue images
    # In LAB, negative b* = blue, positive b* = yellow
    green_b_mean = green_feats[20]  # mean of b* channel
    blue_b_mean = blue_feats[20]
    assert blue_b_mean < green_b_mean
```

**Step 2: Implement color_features.py**

```python
# src/blue_frogs/data/color_features.py
"""LAB color space feature extraction for axanthism detection.

The b* channel in LAB color space encodes the yellow-blue axis:
- Negative b* = blue
- Positive b* = yellow

This directly maps to the biology of axanthism: loss of yellow
xanthophore pigments causes the b* channel to shift negative.
"""

import cv2
import numpy as np


def extract_lab_features(image_rgb: np.ndarray) -> np.ndarray:
    """Extract LAB color space features from an RGB image.

    Returns a 30-dimensional feature vector:
    - 10 features per channel (L, a, b):
      mean, median, std, min, max, Q10, Q25, Q75, Q90, skewness
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    features = []
    for channel_idx in range(3):  # L, a, b
        ch = lab[:, :, channel_idx].ravel()
        features.extend([
            np.mean(ch),
            np.median(ch),
            np.std(ch),
            np.min(ch),
            np.max(ch),
            np.percentile(ch, 10),
            np.percentile(ch, 25),
            np.percentile(ch, 75),
            np.percentile(ch, 90),
            float(_skewness(ch)),
        ])

    return np.array(features, dtype=np.float32)


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of an array."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))
```

**Step 3: Write failing tests for Model B**

```python
# tests/test_model_b.py
"""Tests for Model B: two-stage detection + classification."""

import pytest
import torch
from blue_frogs.models.model_b_classifier import FusionClassifier


def test_fusion_classifier_forward():
    model = FusionClassifier(backbone="tf_efficientnetv2_s", color_feature_dim=30)
    images = torch.randn(2, 3, 384, 384)
    color_feats = torch.randn(2, 30)
    logits = model(images, color_feats)
    assert logits.shape == (2, 1)
```

**Step 4: Implement model_b_detector.py**

```python
# src/blue_frogs/models/model_b_detector.py
"""Model B Stage 1: Frog detection and cropping using YOLOv8 or MegaDetector."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrogDetector:
    """Detect and crop frogs from images using a pretrained detector."""

    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self._model = None
        self._model_path = model_path

    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            if self._model_path:
                self._model = YOLO(self._model_path)
            else:
                # Default: YOLOv8 pretrained on COCO (has 'frog' class)
                self._model = YOLO("yolov8m.pt")

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Detect frogs in an image. Returns list of bounding boxes."""
        self._load_model()
        results = self._model(image_rgb, verbose=False, conf=self.conf_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                # COCO class 'frog' = id 30
                if cls_name == "frog" or cls_id == 30:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf[0]),
                    })
        return detections

    def crop_frog(
        self, image_rgb: np.ndarray, padding_fraction: float = 0.1
    ) -> np.ndarray | None:
        """Detect and crop the most confident frog from the image.

        Returns the cropped image or None if no frog detected.
        Falls back to center crop if detection fails.
        """
        detections = self.detect(image_rgb)
        if not detections:
            # Fallback: center crop (80% of image)
            h, w = image_rgb.shape[:2]
            margin_h, margin_w = int(h * 0.1), int(w * 0.1)
            return image_rgb[margin_h:h - margin_h, margin_w:w - margin_w]

        # Take highest confidence detection
        best = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]

        # Add padding
        h, w = image_rgb.shape[:2]
        pad_h = int((y2 - y1) * padding_fraction)
        pad_w = int((x2 - x1) * padding_fraction)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return image_rgb[y1:y2, x1:x2]
```

**Step 5: Implement model_b_classifier.py**

```python
# src/blue_frogs/models/model_b_classifier.py
"""Model B Stage 2: CNN + LAB color feature fusion classifier."""

import timm
import torch
import torch.nn as nn

from blue_frogs.models.common import BaseClassifier


class FusionClassifier(BaseClassifier):
    """EfficientNetV2-S backbone fused with explicit LAB color features."""

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
        color_feature_dim: int = 30,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        cnn_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(cnn_dim + color_feature_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self, images: torch.Tensor, color_features: torch.Tensor
    ) -> torch.Tensor:
        cnn_features = self.backbone(images)
        fused = torch.cat([cnn_features, color_features], dim=-1)
        return self.head(fused)

    def training_step(self, batch, batch_idx):
        images, color_feats, labels = batch
        logits = self(images, color_feats)
        loss = self.compute_loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, color_feats, labels = batch
        logits = self(images, color_feats)
        loss = self.compute_loss(logits, labels)
        probs = torch.sigmoid(logits.squeeze(-1))
        self._val_outputs.append({
            "loss": loss,
            "probs": probs.detach().cpu(),
            "labels": labels.detach().cpu(),
        })
        return loss
```

**Step 6: Create config and commit**

```yaml
# configs/model_b.yaml
detector:
  model: yolov8m.pt
  conf_threshold: 0.5
  padding_fraction: 0.1

classifier:
  backbone: tf_efficientnetv2_s
  pretrained: true
  color_feature_dim: 30
  fusion_hidden: 256
  dropout: 0.3

training:
  loss_type: weighted_bce
  pos_weight: 20.0
  optimizer: sgd
  learning_rate: 0.001
  weight_decay: 0.0
  max_epochs: 80
  batch_size: 32
  early_stopping_patience: 10
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

data:
  image_size: 384
  n_folds: 5
  seed: 42
```

```bash
pytest tests/test_color_features.py tests/test_model_b.py -v
git add src/blue_frogs/data/color_features.py src/blue_frogs/models/model_b_detector.py \
    src/blue_frogs/models/model_b_classifier.py configs/model_b.yaml \
    tests/test_color_features.py tests/test_model_b.py
git commit -m "feat: Model B - two-stage detector + CNN/LAB fusion classifier"
```

---

## Task 10: Model C - Foundation Models (DINOv2 / BioCLIP)

**Files:**
- Create: `src/blue_frogs/models/model_c.py`
- Create: `configs/model_c.yaml`
- Create: `tests/test_model_c.py`

**Step 1: Write failing tests**

```python
# tests/test_model_c.py
"""Tests for Model C: foundation model classifiers."""

import pytest
import torch
from blue_frogs.models.model_c import FoundationModelClassifier


def test_dinov2_forward_shape():
    model = FoundationModelClassifier(backbone="dinov2", model_size="small")
    x = torch.randn(2, 3, 384, 384)
    logits = model(x)
    assert logits.shape == (2, 1)


def test_linear_probe_freezes_backbone():
    model = FoundationModelClassifier(
        backbone="dinov2", model_size="small", freeze_backbone=True
    )
    backbone_params = list(model.backbone.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    head_params = list(model.head.parameters())
    assert all(p.requires_grad for p in head_params)
```

**Step 2: Implement model_c.py**

```python
# src/blue_frogs/models/model_c.py
"""Model C: Foundation model classifiers (DINOv2 / BioCLIP)."""

import torch
import torch.nn as nn

from blue_frogs.models.common import BaseClassifier


class FoundationModelClassifier(BaseClassifier):
    """Vision foundation model with classification head."""

    def __init__(
        self,
        backbone: str = "dinov2",
        model_size: str = "base",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.backbone_name = backbone

        if backbone == "dinov2":
            self.backbone, feature_dim = self._load_dinov2(model_size)
        elif backbone == "bioclip":
            self.backbone, feature_dim = self._load_bioclip()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _load_dinov2(self, model_size: str):
        """Load DINOv2 from torch hub."""
        model_names = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
        }
        model_name = model_names.get(model_size, "dinov2_vitb14")
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        feature_dim = model.embed_dim
        return model, feature_dim

    def _load_bioclip(self):
        """Load BioCLIP vision encoder."""
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )
        # Extract only the visual encoder
        visual = model.visual
        feature_dim = visual.output_dim
        return visual, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "dinov2":
            features = self.backbone(x)  # CLS token
        elif self.backbone_name == "bioclip":
            features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)

    def configure_optimizers(self):
        """Differential learning rates: low for backbone, high for head."""
        if self.hparams.freeze_backbone:
            return super().configure_optimizers()

        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.learning_rate * 0.01},
            {"params": head_params, "lr": self.learning_rate},
        ], weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
```

**Step 3: Config and commit**

```yaml
# configs/model_c.yaml
model:
  backbone: dinov2  # or "bioclip"
  model_size: base
  hidden_dim: 256
  dropout: 0.3

training:
  loss_type: weighted_bce
  pos_weight: 20.0
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  max_epochs: 80
  batch_size: 16  # ViT needs more memory
  early_stopping_patience: 10
  early_stopping_monitor: val/auprc
  early_stopping_mode: max

linear_probe:
  freeze_backbone: true
  max_epochs: 20
  learning_rate: 0.001

finetune:
  freeze_backbone: false
  backbone_lr_factor: 0.01

data:
  image_size: 384
  n_folds: 5
  seed: 42
```

```bash
pytest tests/test_model_c.py -v
git add src/blue_frogs/models/model_c.py configs/model_c.yaml tests/test_model_c.py
git commit -m "feat: Model C - DINOv2 and BioCLIP foundation model classifiers"
```

---

## Task 11: Model Comparison & Statistical Tests

**Files:**
- Create: `src/blue_frogs/evaluation/comparison.py`
- Create: `scripts/compare_models.py`
- Create: `tests/test_comparison.py`

**Step 1: Write failing tests**

```python
# tests/test_comparison.py
"""Tests for model comparison statistics."""

import numpy as np
from blue_frogs.evaluation.comparison import mcnemar_test, compare_models


def test_mcnemar_test_identical_models():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    preds_a = np.array([0, 0, 1, 1, 0, 1])
    preds_b = np.array([0, 0, 1, 1, 0, 1])
    result = mcnemar_test(y_true, preds_a, preds_b)
    assert result["p_value"] >= 0.05  # no significant difference


def test_compare_models_returns_table():
    y_true = np.array([0] * 50 + [1] * 10)
    rng = np.random.RandomState(42)
    scores = {
        "model_a": rng.random(60),
        "model_b": rng.random(60),
    }
    table = compare_models(y_true, scores)
    assert "model_a" in table
    assert "model_b" in table
    assert "auprc" in table["model_a"]
```

**Step 2: Implement comparison.py**

```python
# src/blue_frogs/evaluation/comparison.py
"""Statistical comparison of model performance."""

import numpy as np
from scipy.stats import chi2

from blue_frogs.evaluation.metrics import (
    compute_classification_metrics,
    compute_bootstrap_ci,
)


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """McNemar's test comparing two classifiers' predictions."""
    # Count discordant pairs
    a_correct_b_wrong = np.sum((preds_a == y_true) & (preds_b != y_true))
    a_wrong_b_correct = np.sum((preds_a != y_true) & (preds_b == y_true))

    n = a_correct_b_wrong + a_wrong_b_correct
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0}

    # McNemar's test with continuity correction
    statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / n
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_discordant": int(n),
        "a_correct_b_wrong": int(a_correct_b_wrong),
        "a_wrong_b_correct": int(a_wrong_b_correct),
    }


def compare_models(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compare multiple models on the same test set.

    Returns a dict of model_name -> metrics (with bootstrap CIs).
    """
    results = {}
    for name, scores in model_scores.items():
        metrics = compute_classification_metrics(y_true, scores)
        for metric_name in ["auroc", "auprc"]:
            ci = compute_bootstrap_ci(y_true, scores, metric_name, n_bootstrap, seed=seed)
            metrics[f"{metric_name}_ci_lower"] = ci["lower"]
            metrics[f"{metric_name}_ci_upper"] = ci["upper"]
        results[name] = metrics
    return results
```

**Step 3: Commit**

```bash
pytest tests/test_comparison.py -v
git add src/blue_frogs/evaluation/comparison.py scripts/compare_models.py tests/test_comparison.py
git commit -m "feat: model comparison with McNemar's test and bootstrap CIs"
```

---

## Task 12: Batch Inference Pipeline

**Files:**
- Create: `src/blue_frogs/inference/batch_scorer.py`
- Create: `scripts/run_inference.py`
- Create: `tests/test_batch_scorer.py`

**Step 1: Write failing tests**

```python
# tests/test_batch_scorer.py
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
```

**Step 2: Implement batch_scorer.py**

```python
# src/blue_frogs/inference/batch_scorer.py
"""Batch inference pipeline for scoring the full iNat frog corpus."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blue_frogs.data.dataset import FrogDataset, get_val_transforms

logger = logging.getLogger(__name__)


def format_prediction_row(
    observation_id: int,
    photo_id: int,
    score: float,
    threshold: float,
    model_version: str,
) -> dict[str, Any]:
    """Format a single prediction as a result row."""
    return {
        "observation_id": observation_id,
        "photo_id": photo_id,
        "prediction_score": round(score, 6),
        "predicted_class": int(score >= threshold),
        "model_version": model_version,
    }


def filter_flagged_predictions(
    df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Filter predictions to those above the flagging threshold."""
    return df[df["prediction_score"] >= threshold].copy()


@torch.no_grad()
def run_batch_inference(
    model,
    metadata: list[dict],
    image_dir: Path,
    threshold: float = 0.5,
    model_version: str = "unknown",
    batch_size: int = 64,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Run inference on a batch of images and return predictions DataFrame."""
    model.eval()
    dataset = FrogDataset(metadata, image_dir, transform=get_val_transforms())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    results = []
    idx = 0
    for images, _ in tqdm(loader, desc="Inference"):
        if torch.cuda.is_available():
            images = images.cuda()
        logits = model(images)
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        for prob in probs:
            entry = metadata[idx]
            results.append(format_prediction_row(
                observation_id=entry["observation_id"],
                photo_id=entry.get("photo_id", 0),
                score=float(prob),
                threshold=threshold,
                model_version=model_version,
            ))
            idx += 1

    return pd.DataFrame(results)
```

**Step 3: Commit**

```bash
pytest tests/test_batch_scorer.py -v
git add src/blue_frogs/inference/batch_scorer.py scripts/run_inference.py tests/test_batch_scorer.py
git commit -m "feat: batch inference pipeline with prediction export"
```

---

## Task 13: Interpretability & Paper Figures

**Files:**
- Create: `src/blue_frogs/evaluation/interpretability.py`
- Create: `src/blue_frogs/figures/paper_figures.py`
- Create: `scripts/generate_figures.py`

**Step 1: Implement interpretability.py**

```python
# src/blue_frogs/evaluation/interpretability.py
"""Model interpretability: Grad-CAM and attention visualization."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM for CNN models (Models A and B)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for the positive class."""
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear")
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
```

**Step 2: Implement paper_figures.py (key figures)**

```python
# src/blue_frogs/figures/paper_figures.py
"""Publication-quality figures for the axanthism classifier paper."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
})


def plot_precision_recall_comparison(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    save_path: str | None = None,
):
    """Plot precision-recall curves for all models on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"model_a": "#1f77b4", "model_b": "#ff7f0e", "model_c": "#2ca02c"}
    labels = {"model_a": "EfficientNetV2-S", "model_b": "Two-Stage + LAB", "model_c": "DINOv2"}

    for name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        auprc = np.trapz(precision, recall)
        ax.plot(
            recall, precision,
            color=colors.get(name, "gray"),
            label=f"{labels.get(name, name)} (AUPRC={auprc:.3f})",
            linewidth=2,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_lab_distributions(
    positive_features: np.ndarray,
    negative_features: np.ndarray,
    save_path: str | None = None,
):
    """Plot LAB b* channel distributions for axanthic vs. normal frogs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channel_names = ["L* (Luminance)", "a* (Green-Red)", "b* (Blue-Yellow)"]
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        offset = i * 10  # 10 features per channel, mean is index 0
        pos_mean = positive_features[:, offset]
        neg_mean = negative_features[:, offset]

        ax.hist(neg_mean, bins=30, alpha=0.5, label="Normal", color="#2ca02c", density=True)
        ax.hist(pos_mean, bins=30, alpha=0.5, label="Axanthic", color="#1f77b4", density=True)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("LAB Color Channel Distributions")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
```

**Step 3: Commit**

```bash
git add src/blue_frogs/evaluation/interpretability.py src/blue_frogs/figures/paper_figures.py \
    scripts/generate_figures.py
git commit -m "feat: Grad-CAM interpretability and publication figure generators"
```

---

## Execution Summary

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Project scaffold & environment | None |
| 2 | Label loading & parsing | Task 1 |
| 3 | iNaturalist API client | Task 1 |
| 4 | Image downloader | Tasks 2, 3 |
| 5 | Training set curation & splits | Task 2 |
| 6 | PyTorch dataset & augmentation | Task 1 |
| 7 | Shared training infra (losses, metrics, base module) | Task 6 |
| 8 | Model A: EfficientNetV2-S ensemble | Task 7 |
| 9 | Model B: two-stage detection + classification | Tasks 7, (color features from 6) |
| 10 | Model C: DINOv2 / BioCLIP | Task 7 |
| 11 | Model comparison & statistical tests | Tasks 8, 9, 10 |
| 12 | Batch inference pipeline | Tasks 8-10 |
| 13 | Interpretability & paper figures | Tasks 11, 12 |

**Parallelizable on HPC:** Tasks 8, 9, 10 can run simultaneously once Tasks 1-7 are complete.
