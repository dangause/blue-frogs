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
