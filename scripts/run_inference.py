"""CLI script to run batch inference on the full iNat frog corpus."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from blue_frogs.config import RESULTS_DIR
from blue_frogs.inference.batch_scorer import run_batch_inference, filter_flagged_predictions
from blue_frogs.models.model_a import EfficientNetClassifier
from blue_frogs.models.model_b_classifier import FusionClassifier
from blue_frogs.models.model_c import FoundationModelClassifier
from blue_frogs.models.model_d import DINOv3Classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "model_a": EfficientNetClassifier,
    "model_b": FusionClassifier,
    "model_c": FoundationModelClassifier,
    "model_d": DINOv3Classifier,
}


def get_model_class(model_name: str):
    """Resolve model name to class."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_name]


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on frog images")
    parser.add_argument("--model", type=str, default="model_a",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model architecture to use")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Directory containing images to score")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="JSON file with image metadata")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-version", type=str, default=None)
    args = parser.parse_args()

    if args.model_version is None:
        args.model_version = f"{args.model}_v1"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_class = get_model_class(args.model)
    logger.info(f"Loading {args.model} from {args.checkpoint}")
    model = model_class.load_from_checkpoint(str(args.checkpoint))

    # Load metadata
    metadata = pd.read_json(args.metadata).to_dict("records")
    logger.info(f"Scoring {len(metadata)} images")

    # Run inference
    predictions = run_batch_inference(
        model=model,
        metadata=metadata,
        image_dir=args.image_dir,
        threshold=args.threshold,
        model_version=args.model_version,
        batch_size=args.batch_size,
    )

    # Save all predictions
    all_path = args.output_dir / f"predictions_{args.model_version}.csv"
    predictions.to_csv(all_path, index=False)
    logger.info(f"All predictions saved to {all_path}")

    # Save flagged predictions
    flagged = filter_flagged_predictions(predictions, args.threshold)
    flagged_path = args.output_dir / f"flagged_{args.model_version}.csv"
    flagged.to_csv(flagged_path, index=False)
    logger.info(f"Flagged {len(flagged)} observations above threshold {args.threshold}")


if __name__ == "__main__":
    main()
