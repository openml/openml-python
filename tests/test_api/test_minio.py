from __future__ import annotations

import pytest
from botocore import UNSIGNED

from openml._api.clients.minio import _make_s3_client, _parse_s3_url

_ENDPOINT = "http://localhost:8000/s3"


def test_make_s3_client(mocker) -> None:
    boto_client = mocker.patch("openml._api.clients.minio.boto3.client")

    _make_s3_client(_ENDPOINT)

    kwargs = boto_client.call_args.kwargs
    assert kwargs["endpoint_url"] == _ENDPOINT
    assert kwargs["region_name"] == "eu-central-1"
    assert kwargs["config"].signature_version == UNSIGNED
    assert kwargs["config"].s3 == {"addressing_style": "path"}


@pytest.mark.parametrize(
    "proxy",
    [
        "http://proxy.example.com:443",
        "https://proxy.example.com:8443",
    ],
)
def test_make_s3_client_with_proxy(mocker, proxy: str) -> None:
    boto_client = mocker.patch("openml._api.clients.minio.boto3.client")

    _make_s3_client(_ENDPOINT, proxy=proxy)

    assert boto_client.call_args.kwargs["config"].proxies == {
        "http": proxy,
        "https": proxy,
    }


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # local reverse-proxy mount
        (
            "http://localhost:8000/s3/datasets/0020/dataset_20.pq",
            ("http://localhost:8000/s3", "datasets", "0020/dataset_20.pq"),
        ),
        # production: no mount
        (
            "https://data.openml.org/datasets/0020/dataset_20.pq",
            ("https://data.openml.org", "datasets", "0020/dataset_20.pq"),
        ),
        # legacy bucket-first layout
        (
            "http://data.openml.org/dataset20/dataset_20.pq",
            ("http://data.openml.org", "dataset20", "dataset_20.pq"),
        ),
    ],
)
def test_parse_s3_url(url: str, expected: tuple[str, str, str]) -> None:
    assert _parse_s3_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/",  # no bucket, no key
        "http://localhost:8000/s3/datasets",  # mount, no key
    ],
)
def test_parse_s3_url_rejects_incomplete(url: str) -> None:
    with pytest.raises(ValueError, match="object key"):
        _parse_s3_url(url)
