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
