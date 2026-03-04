"""Streaming batch inference pipeline for scoring the full iNat frog corpus.

Downloads, scores, and deletes images in bounded batches so disk usage stays
constant regardless of corpus size.  Progress is persisted to a JSON state
file for crash-safe resumability.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from blue_frogs.config import INAT_RATE_LIMIT
from blue_frogs.data.color_features import extract_lab_features
from blue_frogs.data.dataset import FrogDataset, get_val_transforms
from blue_frogs.data.downloader import build_download_manifest, download_batch
from blue_frogs.data.inat_client import fetch_anura_observations_page, get_photo_urls
from blue_frogs.inference.batch_scorer import format_prediction_row

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

@dataclass
class StreamState:
    """Tracks streaming inference progress for resumability."""

    last_id_above: int = 0
    total_observations: int = 0
    total_photos_scored: int = 0
    batches_completed: int = 0
    model: str = ""
    checkpoint: str = ""


def load_state(path: Path) -> StreamState:
    """Load state from *path*, or return a fresh default."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return StreamState(**data)
    return StreamState()


def save_state(path: Path, state: StreamState) -> None:
    """Atomically persist *state* (write-tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Observation fetching
# ---------------------------------------------------------------------------

def fetch_observation_batch(
    id_above: int,
    target_count: int,
    per_page: int = 200,
) -> list[dict[str, Any]]:
    """Accumulate API pages until *target_count* observations collected.

    Returns early when the API returns an empty page (end of corpus).
    Respects iNat rate limits between requests.
    """
    observations: list[dict] = []
    cursor = id_above
    delay = 60.0 / INAT_RATE_LIMIT

    while len(observations) < target_count:
        response = fetch_anura_observations_page(
            id_above=cursor, per_page=per_page,
        )
        results = response.get("results", [])
        if not results:
            break  # end of corpus
        observations.extend(results)
        cursor = max(r["id"] for r in results)
        # Rate-limit between pages
        if len(observations) < target_count:
            time.sleep(delay)

    return observations


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def build_scoring_metadata(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a download manifest into FrogDataset-compatible metadata.

    Each manifest entry has keys: observation_id, photo_id, save_path.
    FrogDataset expects:         observation_id, photo_id, photo_path (relative), label.
    """
    metadata = []
    for entry in manifest:
        save_path = Path(entry["save_path"])
        # photo_path is "obs_id/photo_id.jpg" (two last components)
        photo_path = str(save_path.parent.name / Path(save_path.name))
        metadata.append({
            "observation_id": entry["observation_id"],
            "photo_id": entry["photo_id"],
            "photo_path": photo_path,
            "label": 0,  # unknown at inference time
        })
    return metadata


# ---------------------------------------------------------------------------
# Model B preprocessing
# ---------------------------------------------------------------------------

def preprocess_model_b(
    manifest: list[dict[str, Any]],
    image_dir: Path,
    detector,
) -> np.ndarray:
    """Run YOLO crop + LAB extraction for every image in *manifest*.

    Replaces each downloaded image with its cropped version in-place and
    returns an (N, 30) array of LAB colour features aligned with *manifest*.
    """
    color_features = []
    for entry in manifest:
        img_path = Path(entry["save_path"])
        if not img_path.exists():
            # Missing download — use zeros so the row can still be output
            color_features.append(np.zeros(30, dtype=np.float32))
            continue

        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cropped = detector.crop_frog(rgb)
        if cropped is None:
            cropped = rgb  # fallback: use full image

        # Extract LAB features from cropped image
        color_features.append(extract_lab_features(cropped))

        # Overwrite original with crop so FrogDataset loads the crop
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), cropped_bgr)

    return np.stack(color_features).astype(np.float32)


@torch.no_grad()
def score_batch_model_b(
    model,
    metadata: list[dict],
    color_features: np.ndarray,
    image_dir: Path,
    threshold: float = 0.5,
    model_version: str = "unknown",
    batch_size: int = 64,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Score images with a FusionClassifier that needs (images, color_features)."""
    model.eval()
    device = next(model.parameters()).device

    dataset = FrogDataset(metadata, image_dir, transform=get_val_transforms())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    cf_tensor = torch.from_numpy(color_features)
    results = []
    idx = 0

    for images, _ in loader:
        bs = images.size(0)
        batch_cf = cf_tensor[idx : idx + bs].to(device)
        images = images.to(device)
        logits = model(images, batch_cf)
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


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def append_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """Append *df* to *output_path*, writing the header only on first write."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = not output_path.exists() or output_path.stat().st_size == 0
    df.to_csv(output_path, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Main streaming loop
# ---------------------------------------------------------------------------

def run_streaming_inference(
    model,
    model_name: str,
    checkpoint_path: str,
    threshold: float = 0.5,
    model_version: str = "unknown",
    obs_per_batch: int = 1000,
    inference_batch_size: int = 64,
    output_dir: Path = Path("results/streaming"),
    state_file: Path | None = None,
    tmp_dir: Path | None = None,
    max_workers: int = 8,
    max_batches: int | None = None,
    detector=None,
    num_workers: int = 4,
) -> Path:
    """Streaming inference over the full iNat Anura corpus.

    Returns the path to the predictions CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if state_file is None:
        state_file = output_dir / "state.json"
    state_file = Path(state_file)

    if tmp_dir is None:
        tmp_dir = Path(tempfile.gettempdir()) / "blue_frogs_stream"
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"predictions_{model_version}.csv"

    # Load or create state
    state = load_state(state_file)

    # Validate resume consistency
    if state.batches_completed > 0:
        if state.model and state.model != model_name:
            logger.warning(
                "State file model=%s differs from requested model=%s",
                state.model, model_name,
            )
        if state.checkpoint and state.checkpoint != checkpoint_path:
            logger.warning(
                "State file checkpoint=%s differs from requested checkpoint=%s",
                state.checkpoint, checkpoint_path,
            )

    state.model = model_name
    state.checkpoint = checkpoint_path

    logger.info(
        "Starting streaming inference: model=%s, resume_from_id=%d, "
        "batches_done=%d",
        model_name, state.last_id_above, state.batches_completed,
    )

    batch_num = 0
    while True:
        if max_batches is not None and batch_num >= max_batches:
            logger.info("Reached max_batches=%d, stopping.", max_batches)
            break

        # 1. Fetch observations
        logger.info(
            "Batch %d: fetching observations (id_above=%d)...",
            state.batches_completed + 1, state.last_id_above,
        )
        observations = fetch_observation_batch(
            id_above=state.last_id_above,
            target_count=obs_per_batch,
        )
        if not observations:
            logger.info("No more observations. Corpus complete.")
            break

        # 2. Build download manifest
        batch_image_dir = tmp_dir / f"batch_{state.batches_completed + 1}"
        batch_image_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_download_manifest(observations, batch_image_dir)

        if not manifest:
            # observations had no photos — advance cursor and continue
            state.last_id_above = max(o["id"] for o in observations)
            state.total_observations += len(observations)
            save_state(state_file, state)
            batch_num += 1
            continue

        # 3. Download images
        logger.info("Downloading %d images...", len(manifest))
        dl_stats = download_batch(manifest, max_workers=max_workers)
        logger.info(
            "Download stats: success=%d, failed=%d, skipped=%d",
            dl_stats["success"], dl_stats["failed"], dl_stats["skipped"],
        )

        # 4. Build scoring metadata (only for successfully downloaded images)
        available_manifest = [
            e for e in manifest if Path(e["save_path"]).exists()
        ]
        if not available_manifest:
            logger.warning("Batch %d: no images downloaded, skipping scoring.",
                           state.batches_completed + 1)
            # Clean up and advance
            shutil.rmtree(batch_image_dir, ignore_errors=True)
            state.last_id_above = max(o["id"] for o in observations)
            state.total_observations += len(observations)
            save_state(state_file, state)
            batch_num += 1
            continue

        metadata = build_scoring_metadata(available_manifest)

        # 5. Score batch
        logger.info("Scoring %d images with %s...", len(metadata), model_name)
        try:
            if model_name == "model_b" and detector is not None:
                color_features = preprocess_model_b(
                    available_manifest, batch_image_dir, detector,
                )
                predictions = score_batch_model_b(
                    model=model,
                    metadata=metadata,
                    color_features=color_features,
                    image_dir=batch_image_dir,
                    threshold=threshold,
                    model_version=model_version,
                    batch_size=inference_batch_size,
                    num_workers=num_workers,
                )
            else:
                from blue_frogs.inference.batch_scorer import run_batch_inference

                predictions = run_batch_inference(
                    model=model,
                    metadata=metadata,
                    image_dir=batch_image_dir,
                    threshold=threshold,
                    model_version=model_version,
                    batch_size=inference_batch_size,
                    num_workers=num_workers,
                )
        except Exception:
            logger.exception(
                "Batch %d scoring failed — state NOT advanced.",
                state.batches_completed + 1,
            )
            shutil.rmtree(batch_image_dir, ignore_errors=True)
            raise

        # 6. Append predictions to CSV
        append_predictions(predictions, predictions_path)

        # 7. Update + save state (atomic)
        state.last_id_above = max(o["id"] for o in observations)
        state.total_observations += len(observations)
        state.total_photos_scored += len(predictions)
        state.batches_completed += 1
        save_state(state_file, state)

        # 8. Delete batch images
        shutil.rmtree(batch_image_dir, ignore_errors=True)

        # 9. Log progress
        logger.info(
            "Batch %d complete: %d photos scored | cumulative: %d obs, %d photos",
            state.batches_completed,
            len(predictions),
            state.total_observations,
            state.total_photos_scored,
        )

        batch_num += 1

    logger.info(
        "Streaming inference finished: %d batches, %d observations, %d photos scored",
        state.batches_completed,
        state.total_observations,
        state.total_photos_scored,
    )
    return predictions_path
