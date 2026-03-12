"""CLI entry point for streaming batch inference on the full iNat frog corpus."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from blue_frogs.config import RESULTS_DIR
from blue_frogs.inference.batch_scorer import filter_flagged_predictions
from blue_frogs.inference.stream_scorer import run_streaming_inference
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
        "--model", type=str, default="model_a",
        choices=list(MODEL_CLASSES.keys()),
        help="Model architecture to use",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to model checkpoint (.ckpt)",
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
        "--model-version", type=str, default=None,
        help="Version string for output CSV (default: <model>_v1)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_version is None:
        args.model_version = f"{args.model}_v1"

    # Load model
    model_class = MODEL_CLASSES[args.model]
    logger.info("Loading %s from %s", args.model, args.checkpoint)
    model = model_class.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    model.float()  # ensure float32 for inference
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Using CUDA GPU: %s", torch.cuda.get_device_name())
    elif torch.backends.mps.is_available():
        model = model.to("mps")
        logger.info("Using Apple MPS GPU")
    else:
        logger.info("Using CPU")

    # Load detector for Model B
    detector = None
    if args.model == "model_b":
        from blue_frogs.models.model_b_detector import FrogDetector

        det_path = str(args.detector_checkpoint) if args.detector_checkpoint else None
        detector = FrogDetector(model_path=det_path)
        logger.info("Loaded FrogDetector for Model B")

    # Run streaming inference
    predictions_path = run_streaming_inference(
        model=model,
        model_name=args.model,
        checkpoint_path=str(args.checkpoint),
        threshold=args.threshold,
        model_version=args.model_version,
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
    )

    # Generate flagged predictions summary
    logger.info("Predictions saved to %s", predictions_path)
    if predictions_path.exists():
        all_preds = pd.read_csv(predictions_path)
        flagged = filter_flagged_predictions(all_preds, args.threshold)
        flagged_path = predictions_path.parent / f"flagged_{args.model_version}.csv"
        flagged.to_csv(flagged_path, index=False)
        logger.info(
            "Flagged %d / %d photos (%.2f%%) → %s",
            len(flagged), len(all_preds),
            100 * len(flagged) / max(len(all_preds), 1),
            flagged_path,
        )


if __name__ == "__main__":
    main()
