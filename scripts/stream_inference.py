"""CLI entry point for streaming batch inference on the full iNat frog corpus.

Supports scoring with multiple models in a single pass (download once, score N times).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from blue_frogs.config import RESULTS_DIR
from blue_frogs.inference.batch_scorer import filter_flagged_predictions
from blue_frogs.inference.stream_scorer import run_streaming_inference_multi
from blue_frogs.models.model_a import EfficientNetClassifier
from blue_frogs.models.model_b_classifier import FusionClassifier
from blue_frogs.models.model_c import FoundationModelClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "model_a": EfficientNetClassifier,
    "model_b": FusionClassifier,
    "model_c": FoundationModelClassifier,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming batch inference over the full iNat Anura corpus",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        choices=list(MODEL_CLASSES.keys()),
        help="Model architecture(s) to use (e.g. --models model_a model_b model_c)",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", required=True,
        help="Checkpoint path(s) matching --models order",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--obs-per-batch", type=int, default=1000,
        help="Observations to fetch per streaming batch",
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=64,
        help="GPU batch size for model scoring",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR / "streaming",
        help="Directory for predictions CSV and state file",
    )
    parser.add_argument(
        "--state-file", type=Path, default=None,
        help="Path to state JSON (default: <output-dir>/state.json)",
    )
    parser.add_argument(
        "--tmp-dir", type=Path, default=None,
        help="Temporary directory for batch images (cleaned between batches)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Number of download threads",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--detector-checkpoint", type=Path, default=None,
        help="YOLOv8 checkpoint for Model B frog detection",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Stop after N batches (useful for testing)",
    )
    parser.add_argument(
        "--save-flagged", action="store_true", default=False,
        help="Save flagged images to <output-dir>/flagged_images/",
    )
    parser.add_argument(
        "--calibration-file", type=Path, default=None,
        help="Path to calibration.json with per-model temperature and threshold",
    )
    return parser.parse_args()


def load_model(model_name: str, checkpoint: Path):
    """Load a model checkpoint and move to the best available device."""
    model_class = MODEL_CLASSES[model_name]
    logger.info("Loading %s from %s", model_name, checkpoint)
    model = model_class.load_from_checkpoint(str(checkpoint))
    model.eval()
    model.float()
    if torch.cuda.is_available():
        model = model.cuda()
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    return model


def main() -> None:
    args = parse_args()

    if len(args.models) != len(args.checkpoints):
        raise ValueError(
            f"Got {len(args.models)} models but {len(args.checkpoints)} checkpoints"
        )

    # Load all models
    models = {}
    for name, ckpt in zip(args.models, args.checkpoints):
        models[name] = load_model(name, ckpt)

    if torch.cuda.is_available():
        logger.info("Using CUDA GPU: %s", torch.cuda.get_device_name())
    elif torch.backends.mps.is_available():
        logger.info("Using Apple MPS GPU")
    else:
        logger.info("Using CPU")

    # Load detector for Model B if needed
    detector = None
    if "model_b" in args.models:
        from blue_frogs.models.model_b_detector import FrogDetector

        det_path = str(args.detector_checkpoint) if args.detector_checkpoint else None
        detector = FrogDetector(model_path=det_path)
        logger.info("Loaded FrogDetector for Model B")

    # Load calibration if provided
    calibration = None
    if args.calibration_file:
        import json
        with open(args.calibration_file) as f:
            calibration = json.load(f)
        logger.info("Loaded calibration from %s", args.calibration_file)
        for name, cal in calibration.items():
            logger.info("  %s: T=%.3f, threshold=%.4f",
                        name, cal.get("temperature", 1.0), cal.get("threshold", 0.5))

    # Run streaming multi-model inference
    predictions_paths = run_streaming_inference_multi(
        models=models,
        threshold=args.threshold,
        obs_per_batch=args.obs_per_batch,
        inference_batch_size=args.inference_batch_size,
        output_dir=args.output_dir,
        state_file=args.state_file,
        tmp_dir=args.tmp_dir,
        max_workers=args.max_workers,
        max_batches=args.max_batches,
        detector=detector,
        num_workers=args.num_workers,
        save_flagged=args.save_flagged,
        calibration=calibration,
    )

    # Generate flagged predictions summary per model
    for model_name, pred_path in predictions_paths.items():
        if pred_path.exists():
            all_preds = pd.read_csv(pred_path)
            cal = (calibration or {}).get(model_name, {})
            flag_thresh = cal.get("threshold", args.threshold)
            flagged = filter_flagged_predictions(all_preds, flag_thresh)
            flagged_path = pred_path.parent / f"flagged_{model_name}_v1.csv"
            flagged.to_csv(flagged_path, index=False)
            logger.info(
                "%s: Flagged %d / %d photos (%.2f%%) at threshold %.4f → %s",
                model_name, len(flagged), len(all_preds),
                100 * len(flagged) / max(len(all_preds), 1),
                flag_thresh, flagged_path,
            )


if __name__ == "__main__":
    main()
