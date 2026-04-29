"""Tests for iNaturalist API client."""

import pytest
from unittest.mock import patch, MagicMock
from blue_frogs.data.inat_client import (
    get_photo_urls,
    build_medium_url,
    fetch_observation_metadata,
)


def test_build_medium_url_from_square():
    square_url = "https://static.inaturalist.org/photos/61482854/square.jpg?1581761020"
    result = build_medium_url(square_url)
    assert "medium" in result
    assert "square" not in result


def test_build_medium_url_from_other_size():
    url = "https://static.inaturalist.org/photos/61482854/small.jpg"
    result = build_medium_url(url)
    assert "/medium." in result


def test_get_photo_urls_extracts_all_photos():
    observation = {
        "id": 12345,
        "observation_photos": [
            {"photo": {"id": 111, "url": "https://static.inaturalist.org/photos/111/square.jpg"}},
            {"photo": {"id": 222, "url": "https://static.inaturalist.org/photos/222/square.jpg"}},
        ],
    }
    urls = get_photo_urls(observation)
    assert len(urls) == 2
    assert all("medium" in u for u in urls)


def test_get_photo_urls_empty_observation():
    observation = {"id": 12345, "observation_photos": []}
    assert get_photo_urls(observation) == []


@patch("blue_frogs.data.inat_client.get_observations")
def test_fetch_observation_metadata(mock_get):
    mock_get.return_value = {
        "results": [
            {
                "id": 12345,
                "quality_grade": "research",
                "taxon": {"id": 1234, "name": "Hyla cinerea"},
                "observed_on": "2021-06-15",
                "location": "37.5,-77.4",
                "observation_photos": [
                    {"photo": {"id": 111, "url": "https://example.com/photos/111/square.jpg"}}
                ],
            }
        ]
    }
    result = fetch_observation_metadata([12345])
    assert len(result) == 1
    assert result[0]["id"] == 12345
