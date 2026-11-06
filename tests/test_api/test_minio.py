from __future__ import annotations

import pytest

from openml._api.clients.minio import _parse_s3_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "http://object-storage:8333/datasets/0000/0061/dataset_61.pq",
            ("http://object-storage:8333", "datasets", "0000/0061/dataset_61.pq"),
        ),
        (
            "http://localhost:8000/s3/datasets/0000/0061/dataset_61.pq",
            ("http://localhost:8000/s3", "datasets", "0000/0061/dataset_61.pq"),
        ),
        (
            "https://data.openml.org/datasets/0000/0061/dataset_61.pq",
            ("https://data.openml.org", "datasets", "0000/0061/dataset_61.pq"),
        ),
        (
            "http://data.openml.org/dataset61/dataset_61.pq",
            ("http://data.openml.org", "dataset61", "dataset_61.pq"),
        ),
        (
            "http://localhost:8000/minio/datasets/0000/0001/phpFsFYVN",
            ("http://localhost:8000/minio", "datasets", "0000/0001/phpFsFYVN"),
        ),
    ],
)
def test_parse_s3_url(url: str, expected: tuple[str, str, str]) -> None:
    assert _parse_s3_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/",
        "http://localhost:8000/datasets",
        "http://localhost:8000/s3/datasets",
    ],
)
def test_parse_s3_url_rejects_incomplete(url: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        _parse_s3_url(url)
