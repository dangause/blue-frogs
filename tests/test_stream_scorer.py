"""Tests for the streaming batch inference pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from blue_frogs.inference.stream_scorer import (
    StreamState,
    append_predictions,
    build_scoring_metadata,
    fetch_observation_batch,
    load_state,
    preprocess_model_b,
    run_streaming_inference,
    save_state,
    score_batch_model_b,
)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestStateManagement:

    def test_load_state_missing_file(self, tmp_path):
        state = load_state(tmp_path / "nonexistent.json")
        assert state.last_id_above == 0
        assert state.batches_completed == 0

    def test_load_save_roundtrip(self, tmp_path):
        path = tmp_path / "state.json"
        state = StreamState(
            last_id_above=42000,
            total_observations=500,
            total_photos_scored=1200,
            batches_completed=5,
            model="model_a",
            checkpoint="models/model_a.ckpt",
        )
        save_state(path, state)
        loaded = load_state(path)

        assert loaded.last_id_above == 42000
        assert loaded.total_observations == 500
        assert loaded.total_photos_scored == 1200
        assert loaded.batches_completed == 5
        assert loaded.model == "model_a"
        assert loaded.checkpoint == "models/model_a.ckpt"

    def test_save_state_atomic(self, tmp_path):
        """Save should write to .tmp then replace — no .tmp left behind."""
        path = tmp_path / "state.json"
        save_state(path, StreamState(last_id_above=1))
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()

    def test_save_state_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "state.json"
        save_state(path, StreamState())
        assert path.exists()


# ---------------------------------------------------------------------------
# Metadata conversion
# ---------------------------------------------------------------------------

class TestBuildScoringMetadata:

    def test_basic_conversion(self, tmp_path):
        manifest = [
            {
                "observation_id": 100,
                "photo_id": 200,
                "url": "https://example.com/photo.jpg",
                "save_path": tmp_path / "100" / "200.jpg",
            },
            {
                "observation_id": 101,
                "photo_id": 201,
                "url": "https://example.com/photo2.jpg",
                "save_path": tmp_path / "101" / "201.jpg",
            },
        ]
        metadata = build_scoring_metadata(manifest)

        assert len(metadata) == 2
        assert metadata[0]["observation_id"] == 100
        assert metadata[0]["photo_id"] == 200
        assert metadata[0]["photo_path"] == "100/200.jpg"
        assert metadata[0]["label"] == 0
        assert metadata[1]["photo_path"] == "101/201.jpg"


# ---------------------------------------------------------------------------
# CSV appending
# ---------------------------------------------------------------------------

class TestAppendPredictions:

    def test_header_on_first_write(self, tmp_path):
        path = tmp_path / "preds.csv"
        df = pd.DataFrame({
            "observation_id": [1],
            "photo_id": [10],
            "prediction_score": [0.9],
            "predicted_class": [1],
            "model_version": ["v1"],
        })
        append_predictions(df, path)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "observation_id,photo_id,prediction_score,predicted_class,model_version"
        assert len(lines) == 2

    def test_no_header_on_append(self, tmp_path):
        path = tmp_path / "preds.csv"
        df = pd.DataFrame({
            "observation_id": [1],
            "photo_id": [10],
            "prediction_score": [0.9],
            "predicted_class": [1],
            "model_version": ["v1"],
        })
        append_predictions(df, path)
        append_predictions(df, path)
        content = path.read_text()
        lines = content.strip().split("\n")
        # 1 header + 2 data rows
        assert len(lines) == 3
        assert lines[0].startswith("observation_id")

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "preds.csv"
        df = pd.DataFrame({"a": [1]})
        append_predictions(df, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Observation fetching
# ---------------------------------------------------------------------------

class TestFetchObservationBatch:

    @patch("blue_frogs.inference.stream_scorer.fetch_anura_observations_page")
    @patch("blue_frogs.inference.stream_scorer.time.sleep")
    def test_accumulates_pages(self, mock_sleep, mock_fetch):
        """Should accumulate pages until target_count is reached."""
        mock_fetch.side_effect = [
            {"results": [{"id": i} for i in range(1, 6)]},     # 5 obs
            {"results": [{"id": i} for i in range(6, 11)]},    # 5 obs
            {"results": [{"id": i} for i in range(11, 16)]},   # 5 obs
        ]
        observations = fetch_observation_batch(id_above=0, target_count=12, per_page=5)
        assert len(observations) == 15  # 3 full pages
        assert mock_fetch.call_count == 3

    @patch("blue_frogs.inference.stream_scorer.fetch_anura_observations_page")
    @patch("blue_frogs.inference.stream_scorer.time.sleep")
    def test_stops_on_empty_page(self, mock_sleep, mock_fetch):
        """Should stop when API returns empty results (end of corpus)."""
        mock_fetch.side_effect = [
            {"results": [{"id": 1}, {"id": 2}]},
            {"results": []},
        ]
        observations = fetch_observation_batch(id_above=0, target_count=100, per_page=5)
        assert len(observations) == 2

    @patch("blue_frogs.inference.stream_scorer.fetch_anura_observations_page")
    @patch("blue_frogs.inference.stream_scorer.time.sleep")
    def test_cursor_advances(self, mock_sleep, mock_fetch):
        """id_above should advance to max ID of previous page."""
        mock_fetch.side_effect = [
            {"results": [{"id": 10}, {"id": 20}]},
            {"results": [{"id": 30}]},
            {"results": []},  # end of corpus
        ]
        fetch_observation_batch(id_above=0, target_count=100, per_page=5)
        # Second call should use id_above=20 (max from first page)
        assert mock_fetch.call_args_list[1][1]["id_above"] == 20


# ---------------------------------------------------------------------------
# Model B preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessModelB:

    def test_crops_and_extracts_features(self, tmp_path):
        """Should crop image and return 30-D LAB features."""
        # Create a test image
        img_dir = tmp_path / "100"
        img_dir.mkdir()
        img_path = img_dir / "200.jpg"

        import cv2
        green = np.full((100, 100, 3), (0, 128, 0), dtype=np.uint8)
        cv2.imwrite(str(img_path), green)

        manifest = [{
            "observation_id": 100,
            "photo_id": 200,
            "save_path": img_path,
        }]

        detector = MagicMock()
        # Return a cropped region
        detector.crop_frog.return_value = np.full((50, 50, 3), (0, 128, 0), dtype=np.uint8)

        features = preprocess_model_b(manifest, tmp_path, detector)
        assert features.shape == (1, 30)
        assert features.dtype == np.float32

    def test_missing_image_returns_zeros(self, tmp_path):
        manifest = [{
            "observation_id": 999,
            "photo_id": 888,
            "save_path": tmp_path / "nonexistent.jpg",
        }]
        detector = MagicMock()
        features = preprocess_model_b(manifest, tmp_path, detector)
        assert features.shape == (1, 30)
        np.testing.assert_array_equal(features[0], np.zeros(30))


# ---------------------------------------------------------------------------
# Model B scoring
# ---------------------------------------------------------------------------

class TestScoreBatchModelB:

    def test_two_input_forward(self, tmp_path):
        """Verify model receives both images and color_features."""
        # Create mock image
        img_dir = tmp_path / "100"
        img_dir.mkdir()
        img_path = img_dir / "200.jpg"

        from PIL import Image
        Image.new("RGB", (100, 100), color=(0, 128, 0)).save(img_path)

        metadata = [{
            "observation_id": 100,
            "photo_id": 200,
            "photo_path": "100/200.jpg",
            "label": 0,
        }]
        color_features = np.random.randn(1, 30).astype(np.float32)

        # Mock model: verify it receives two inputs
        model = MagicMock()
        model.eval.return_value = None
        params = [torch.tensor([1.0])]
        model.parameters.return_value = iter(params)
        model.return_value = torch.tensor([[0.5]])  # logit

        df = score_batch_model_b(
            model=model,
            metadata=metadata,
            color_features=color_features,
            image_dir=tmp_path,
            threshold=0.5,
            model_version="test_v1",
            batch_size=1,
            num_workers=0,
        )

        assert len(df) == 1
        assert "observation_id" in df.columns
        assert "prediction_score" in df.columns
        # Verify model was called with two args (images, color_features)
        call_args = model.call_args
        assert len(call_args[0]) == 2  # positional: images, color_features


# ---------------------------------------------------------------------------
# Streaming loop integration
# ---------------------------------------------------------------------------

class TestStreamingLoop:

    @patch("blue_frogs.inference.stream_scorer.fetch_observation_batch")
    @patch("blue_frogs.inference.stream_scorer.build_download_manifest")
    @patch("blue_frogs.inference.stream_scorer.download_batch")
    def test_resume_from_state(
        self, mock_download, mock_manifest, mock_fetch, tmp_path,
    ):
        """After saving state, resuming should continue from last_id_above."""
        state_file = tmp_path / "state.json"
        save_state(state_file, StreamState(
            last_id_above=5000,
            total_observations=100,
            total_photos_scored=250,
            batches_completed=1,
            model="model_a",
            checkpoint="test.ckpt",
        ))

        # Return one batch (no photos) then empty
        mock_fetch.side_effect = [
            [{"id": 5001, "observation_photos": []}],
            [],
        ]
        mock_manifest.return_value = []  # no photos → skip scoring

        model = MagicMock()

        run_streaming_inference(
            model=model,
            model_name="model_a",
            checkpoint_path="test.ckpt",
            output_dir=tmp_path / "out",
            state_file=state_file,
            tmp_dir=tmp_path / "tmp",
            max_batches=2,
        )

        # Verify fetch was called starting from 5000
        first_call = mock_fetch.call_args_list[0]
        assert first_call[1].get("id_above", first_call[0][0] if first_call[0] else None) == 5000

        # State should be updated — cursor advanced, observations counted.
        # batches_completed stays at 1 because the batch had no photos to score.
        final_state = load_state(state_file)
        assert final_state.last_id_above == 5001
        assert final_state.total_observations == 101

    @patch("blue_frogs.inference.stream_scorer.fetch_observation_batch")
    def test_stops_on_empty_corpus(self, mock_fetch, tmp_path):
        mock_fetch.return_value = []
        model = MagicMock()

        result = run_streaming_inference(
            model=model,
            model_name="model_a",
            checkpoint_path="test.ckpt",
            output_dir=tmp_path / "out",
            state_file=tmp_path / "state.json",
            tmp_dir=tmp_path / "tmp",
        )
        assert result == tmp_path / "out" / "predictions_unknown.csv"

    @patch("blue_frogs.inference.stream_scorer.fetch_observation_batch")
    def test_max_batches_limit(self, mock_fetch, tmp_path):
        """max_batches should stop the loop even if more data exists."""
        mock_fetch.return_value = []
        model = MagicMock()

        run_streaming_inference(
            model=model,
            model_name="model_a",
            checkpoint_path="test.ckpt",
            output_dir=tmp_path / "out",
            state_file=tmp_path / "state.json",
            tmp_dir=tmp_path / "tmp",
            max_batches=0,
        )
        # With max_batches=0, fetch should never be called
        mock_fetch.assert_not_called()
