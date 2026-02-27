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
