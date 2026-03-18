"""Inference modules for scoring frog images."""

from blue_frogs.inference.batch_scorer import (
    filter_flagged_predictions,
    format_prediction_row,
    run_batch_inference,
)
from blue_frogs.inference.stream_scorer import (
    StreamState,
    append_predictions,
    build_scoring_metadata,
    fetch_observation_batch,
    load_state,
    preprocess_model_b,
    run_streaming_inference,
    run_streaming_inference_multi,
    save_state,
    score_batch_model_b,
)

__all__ = [
    # batch_scorer
    "filter_flagged_predictions",
    "format_prediction_row",
    "run_batch_inference",
    # stream_scorer
    "StreamState",
    "append_predictions",
    "build_scoring_metadata",
    "fetch_observation_batch",
    "load_state",
    "preprocess_model_b",
    "run_streaming_inference",
    "run_streaming_inference_multi",
    "save_state",
    "score_batch_model_b",
]
