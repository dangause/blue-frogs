"""Tests for image downloader."""

import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path
from blue_frogs.data.downloader import (
    build_download_manifest,
    download_image_sync,
)


def test_build_download_manifest():
    observations = [
        {
            "id": 123,
            "observation_photos": [
                {"photo": {"id": 111, "url": "https://example.com/photos/111/square.jpg"}},
                {"photo": {"id": 222, "url": "https://example.com/photos/222/square.jpg"}},
            ],
        },
    ]
    manifest = build_download_manifest(observations, Path("/tmp/images"))
    assert len(manifest) == 2
    assert manifest[0]["photo_id"] == 111
    assert manifest[0]["save_path"] == Path("/tmp/images/123/111.jpg")
    assert "medium" in manifest[0]["url"]


@patch("blue_frogs.data.downloader.requests.get")
def test_download_image_sync(mock_get, tmp_path):
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.content = b"fake image data"
    mock_response.raise_for_status = lambda: None

    save_path = tmp_path / "test.jpg"
    result = download_image_sync("https://example.com/photo.jpg", save_path)
    assert result is True
    assert save_path.read_bytes() == b"fake image data"
