from requests import Response, Request, Session
from unittest.mock import patch
import pytest
import os
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from openml.enums import APIVersion
from openml.exceptions import OpenMLAuthenticationError
from openml._api import HTTPClient, HTTPCache
import openml


@pytest.fixture
def cache(http_client_v1) -> HTTPCache:
    return http_client_v1.cache


@pytest.fixture
def http_client(http_client_v1) -> HTTPClient:
    return http_client_v1


@pytest.fixture
def sample_path() -> str:
    return "task/1"


@pytest.fixture
def sample_url_v1(sample_path, test_server_v1) -> str:
    return urljoin(test_server_v1, sample_path)


@pytest.fixture
def sample_download_url_v1(test_server_v1) -> str:
    server = test_server_v1.split("api/")[0]
    endpoint = "data/v1/download/1/anneal.arff"
    url = server + endpoint
    return url


def test_cache(cache, sample_url_v1):
    params = {"param1": "value1", "param2": "value2"}

    parsed_url = urlparse(sample_url_v1)
    netloc_parts = parsed_url.netloc.split(".")[::-1]
    path_parts = parsed_url.path.strip("/").split("/")
    params_key = "&".join([f"{k}={v}" for k, v in params.items()])


    key = cache.get_key(sample_url_v1, params)

    expected_key = os.path.join(
        *netloc_parts,
        *path_parts,
        params_key,
    )

    assert key == expected_key

    # mock response
    req = Request("GET", sample_url_v1).prepare()
    response = Response()
    response.status_code = 200
    response.url = sample_url_v1
    response.reason = "OK"
    response._content = b"<xml>test</xml>"
    response.headers = {"Content-Type": "text/xml"}
    response.encoding = "utf-8"
    response.request = req
    response.elapsed = type("Elapsed", (), {"total_seconds": lambda x: 0.1})()

    cache.save(key, response)
    cached = cache.load(key)

    assert cached.status_code == 200
    assert cached.url == sample_url_v1
    assert cached.content == b"<xml>test</xml>"
    assert cached.headers["Content-Type"] == "text/xml"


@pytest.mark.test_server()
def test_get(http_client):
    response = http_client.get("task/1")

    assert response.status_code == 200
    assert b"<oml:task" in response.content


@pytest.mark.test_server()
def test_get_with_cache_creates_cache(http_client, cache, sample_url_v1, sample_path):
    response = http_client.get(sample_path, enable_cache=True)

    assert response.status_code == 200
    assert cache.path.exists()

    cache_key = cache.get_key(sample_url_v1, {})
    cache_path = cache._key_to_path(cache_key)

    assert (cache_path / "meta.json").exists()
    assert (cache_path / "headers.json").exists()
    assert (cache_path / "body.bin").exists()


@pytest.mark.test_server()
def test_get_uses_cached_response(http_client, cache, sample_url_v1, sample_path, monkeypatch):
    response = Response()
    response.status_code = 200
    response._content = b"cached-response"
    response.headers = {}

    key = cache.get_key(url=sample_url_v1, params={})
    cache.save(key=key, response=response)

    def fail_request(*args, **kwargs):
        raise AssertionError("HTTP request should not be called")
    monkeypatch.setattr(Session, "request", fail_request)

    cached_response = http_client.get(sample_path, enable_cache=True)

    assert cached_response.status_code == response.status_code
    assert cached_response.content == response.content

@pytest.mark.test_server()
def test_get_refresh_cache(http_client, cache, sample_url_v1, sample_path):
    key = cache.get_key(sample_url_v1, {})
    meta_path = cache._key_to_path(key) / "meta.json"

    r1 = http_client.get(sample_path, enable_cache=True)
    mtime1 = meta_path.stat().st_mtime

    r2 = http_client.get(sample_path, enable_cache=True, refresh_cache=True)
    mtime2 = meta_path.stat().st_mtime

    assert mtime1 != mtime2
    assert r2.status_code == 200
    assert r1.content == r2.content


@pytest.mark.test_server()
def test_get_with_api_key(http_client, sample_path, test_apikey_v1):
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200

        http_client.get(sample_path, use_api_key=True)

        _, kwargs = mock_request.call_args
        assert kwargs.get("params", {}).get("api_key") == test_apikey_v1


@pytest.mark.test_server()
def test_get_without_api_key_raises(http_client):
    with openml.config.overwrite_config_context({"apikey": None}), pytest.raises(OpenMLAuthenticationError):
        http_client.get("task/1", use_api_key=True)


@pytest.mark.test_server()
def test_download_creates_file(http_client, sample_download_url_v1):
    dummy_content = b"this is dummy content"
    md5_checksum = hashlib.md5(dummy_content).hexdigest()

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = dummy_content

        path = http_client.download(
            url=sample_download_url_v1,
            file_name="downloaded.arff",
            md5_checksum=md5_checksum,
        )

    assert path.exists()
    assert path.is_file()
    assert path.read_bytes() == dummy_content


@pytest.mark.test_server()
def test_download_is_cached_on_disk(http_client, sample_download_url_v1):
    path1 = http_client.download(
        url=sample_download_url_v1,
        file_name="cached.arff",
    )
    mtime1 = path1.stat().st_mtime

    path2 = http_client.download(
        url=sample_download_url_v1,
        file_name="cached.arff",
    )
    mtime2 = path2.stat().st_mtime

    assert path1 == path2
    assert mtime1 == mtime2


@pytest.mark.test_server()
def test_download_respects_custom_handler(http_client, sample_download_url_v1):
    def handler(response, path: Path, encoding: str):
        path.write_text("HANDLED", encoding=encoding)
        return path

    path = http_client.download(
        url=sample_download_url_v1,
        file_name="handler.arff",
        handler=handler,
    )

    assert path.exists()
    assert path.read_text() == "HANDLED"


def test_post(http_client, test_server_v1, test_apikey_v1):
    resource_name = "resource"
    resource_files = {"description": "Resource Description File"}

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200

        http_client.post(resource_name, files=resource_files)

        mock_request.assert_called_once_with(
            method="POST",
            url=urljoin(test_server_v1, resource_name),
            params={},
            data={"api_key": test_apikey_v1},
            headers=openml.config._HEADERS,
            files=resource_files,
        )


def test_delete(http_client, test_server_v1, test_apikey_v1):
    resource_name = "resource"
    resource_id = 123

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200

        http_client.delete(f"{resource_name}/{resource_id}")

        mock_request.assert_called_once_with(
            method="DELETE",
            url=(
                test_server_v1
                + resource_name
                + "/"
                + str(resource_id)
            ),
            params={"api_key": test_apikey_v1},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )
