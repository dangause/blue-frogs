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
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Run inference on a batch of images and return predictions DataFrame.

    Args:
        temperature: Temperature scaling parameter. Values > 1 soften
            overconfident predictions (from calibration).
    """
    model.eval()
    dataset = FrogDataset(metadata, image_dir, transform=get_val_transforms())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    device = next(model.parameters()).device

    results = []
    idx = 0
    for images, _ in tqdm(loader, desc="Inference"):
        images = images.to(device)
        logits = model(images).squeeze(-1)
        probs = torch.sigmoid(logits / temperature).cpu().numpy()

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
