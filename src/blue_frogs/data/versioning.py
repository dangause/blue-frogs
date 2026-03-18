"""Dataset versioning and integrity verification.

Provides SHA256 hashing of dataset files for reproducibility verification.
Stores hashes in data/dataset_hashes.json.

Usage:
    from blue_frogs.data.versioning import compute_dataset_hash, verify_dataset_hash

    # Compute and save hash
    hash_info = compute_dataset_hash(Path("data/"))
    save_dataset_hash(hash_info, Path("data/dataset_hashes.json"))

    # Verify integrity
    is_valid = verify_dataset_hash(Path("data/"), expected_hash)
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Key files to hash for dataset versioning
KEY_FILES = [
    "labels.json",
    "splits/train_test_split.json",
    "curated/positive_metadata.json",
    "curated/negative_metadata.json",
]


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a single file."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_hash(
    data_dir: Path,
    key_files: list[str] | None = None,
    algorithm: str = "sha256",
) -> dict:
    """Compute combined hash of key dataset files.

    Args:
        data_dir: Root data directory
        key_files: List of relative paths to hash (uses KEY_FILES if None)
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Dict with:
            - combined_hash: Hash of concatenated file hashes
            - file_hashes: Individual file hashes
            - files_found: Number of files found
            - files_missing: List of missing files
            - timestamp: ISO timestamp
    """
    if key_files is None:
        key_files = KEY_FILES

    file_hashes = {}
    files_missing = []

    for rel_path in sorted(key_files):
        file_path = data_dir / rel_path
        if file_path.exists():
            file_hashes[rel_path] = compute_file_hash(file_path, algorithm)
        else:
            files_missing.append(rel_path)
            logger.warning(f"File not found for hashing: {file_path}")

    # Compute combined hash from sorted file hashes
    combined = hashlib.new(algorithm)
    for rel_path in sorted(file_hashes.keys()):
        combined.update(f"{rel_path}:{file_hashes[rel_path]}\n".encode())

    return {
        "combined_hash": combined.hexdigest(),
        "algorithm": algorithm,
        "file_hashes": file_hashes,
        "files_found": len(file_hashes),
        "files_missing": files_missing,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def save_dataset_hash(hash_info: dict, output_path: Path) -> None:
    """Save dataset hash info to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hash_info, f, indent=2)
    logger.info(f"Dataset hash saved to {output_path}")


def load_dataset_hash(hash_path: Path) -> dict | None:
    """Load dataset hash info from JSON file."""
    if not hash_path.exists():
        return None
    with open(hash_path) as f:
        return json.load(f)


def verify_dataset_hash(
    data_dir: Path,
    expected_hash: str,
    key_files: list[str] | None = None,
) -> bool:
    """Verify dataset integrity against expected hash.

    Args:
        data_dir: Root data directory
        expected_hash: Expected combined hash
        key_files: List of relative paths to hash

    Returns:
        True if hash matches, False otherwise
    """
    current = compute_dataset_hash(data_dir, key_files)
    matches = current["combined_hash"] == expected_hash

    if not matches:
        logger.warning(
            f"Dataset hash mismatch: expected {expected_hash[:16]}..., "
            f"got {current['combined_hash'][:16]}..."
        )

    return matches


def verify_against_file(data_dir: Path, hash_path: Path) -> dict:
    """Verify dataset against stored hash file.

    Returns dict with verification result and details.
    """
    stored = load_dataset_hash(hash_path)
    if stored is None:
        return {
            "valid": False,
            "error": "Hash file not found",
            "hash_path": str(hash_path),
        }

    current = compute_dataset_hash(
        data_dir,
        key_files=list(stored.get("file_hashes", {}).keys()) or None,
    )

    # Check combined hash
    combined_match = current["combined_hash"] == stored["combined_hash"]

    # Check individual files
    file_mismatches = []
    for rel_path, expected in stored.get("file_hashes", {}).items():
        current_hash = current["file_hashes"].get(rel_path)
        if current_hash != expected:
            file_mismatches.append({
                "file": rel_path,
                "expected": expected[:16] + "...",
                "actual": (current_hash[:16] + "...") if current_hash else "MISSING",
            })

    return {
        "valid": combined_match and not file_mismatches,
        "combined_hash_match": combined_match,
        "file_mismatches": file_mismatches,
        "files_checked": current["files_found"],
        "stored_timestamp": stored.get("timestamp"),
        "current_timestamp": current["timestamp"],
    }
