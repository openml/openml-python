from __future__ import annotations

import unittest.mock
from pathlib import Path
from typing import NamedTuple, Iterable, Iterator
from unittest import mock

import minio
import pytest
import requests

import openml
from openml.config import ConfigurationForExamples
import openml.testing
from openml._api_calls import _download_minio_bucket, API_TOKEN_HELP_LINK


class TestConfig(openml.testing.TestBase):
    @pytest.mark.test_server()
    def test_too_long_uri(self):
        with pytest.raises(openml.exceptions.OpenMLServerError, match="URI too long!"):
            openml.datasets.list_datasets(data_id=list(range(10000)))

    @unittest.mock.patch("time.sleep")
    @unittest.mock.patch("requests.Session")
    @pytest.mark.test_server()
    def test_retry_on_database_error(self, Session_class_mock, _):
        response_mock = unittest.mock.Mock()
        response_mock.text = (
            "<oml:error>\n"
            "<oml:code>107</oml:code>"
            "<oml:message>Database connection error. "
            "Usually due to high server load. "
            "Please wait for N seconds and try again.</oml:message>\n"
            "</oml:error>"
        )
        Session_class_mock.return_value.__enter__.return_value.get.return_value = response_mock
        with pytest.raises(openml.exceptions.OpenMLServerException, match="/abc returned code 107"):
            openml._api_calls._send_request("get", "/abc", {})

        assert Session_class_mock.return_value.__enter__.return_value.get.call_count == 20


class FakeObject(NamedTuple):
    object_name: str
    etag: str
    """We use the etag of a Minio object as the name of a marker if we already downloaded it."""


class FakeMinio:
    def __init__(self, objects: Iterable[FakeObject] | None = None):
        self._objects = objects or []

    def list_objects(self, *args, **kwargs) -> Iterator[FakeObject]:
        yield from self._objects

    def fget_object(self, object_name: str, file_path: str, *args, **kwargs) -> None:
        if object_name in [obj.object_name for obj in self._objects]:
            Path(file_path).write_text("foo")
            return
        raise FileNotFoundError


@mock.patch.object(minio, "Minio")
def test_download_all_files_observes_cache(mock_minio, tmp_path: Path) -> None:
    some_prefix, some_filename = "some/prefix", "dataset.arff"
    some_object_path = f"{some_prefix}/{some_filename}"
    some_url = f"https://not.real.com/bucket/{some_object_path}"
    mock_minio.return_value = FakeMinio(
        objects=[
            FakeObject(object_name=some_object_path, etag=str(hash(some_object_path))),
        ],
    )

    _download_minio_bucket(source=some_url, destination=tmp_path)
    time_created = (tmp_path / "dataset.arff").stat().st_ctime

    _download_minio_bucket(source=some_url, destination=tmp_path)
    time_modified = (tmp_path / some_filename).stat().st_mtime

    assert time_created == time_modified


@mock.patch.object(minio, "Minio")
def test_download_minio_failure(mock_minio, tmp_path: Path) -> None:
    some_prefix, some_filename = "some/prefix", "dataset.arff"
    some_object_path = f"{some_prefix}/{some_filename}"
    some_url = f"https://not.real.com/bucket/{some_object_path}"
    mock_minio.return_value = FakeMinio(
        objects=[
            FakeObject(object_name=None, etag="tmp"),
        ],
    )

    with pytest.raises(ValueError):
        _download_minio_bucket(source=some_url, destination=tmp_path)

    mock_minio.return_value = FakeMinio(
        objects=[
            FakeObject(object_name="tmp", etag=None),
        ],
    )

    with pytest.raises(ValueError):
        _download_minio_bucket(source=some_url, destination=tmp_path)


@pytest.mark.parametrize(
    "endpoint, method",
    [
        # https://github.com/openml/OpenML/blob/develop/openml_OS/views/pages/api_new/v1/xml/pre.php
        ("flow/exists", "post"),  # 102
        ("dataset", "post"),  # 137
        ("dataset/42", "delete"),  # 350
        # ("flow/owned", "post"),  # 310 - Couldn't find what would trigger this
        ("flow/42", "delete"),  # 320
        ("run/42", "delete"),  # 400
        ("task/42", "delete"),  # 460
    ],
)
@pytest.mark.test_server()
def test_authentication_endpoints_requiring_api_key_show_relevant_help_link(
    endpoint: str,
    method: str,
) -> None:
    # We need to temporarily disable the API key to test the error message
    with openml.config.overwrite_config_context({"apikey": None}):
        with pytest.raises(openml.exceptions.OpenMLAuthenticationError, match=API_TOKEN_HELP_LINK):
            openml._api_calls._perform_api_call(call=endpoint, request_method=method, data=None)


def _make_ok_response() -> unittest.mock.Mock:
    """Return a minimal mock that passes __check_response without errors."""
    response = unittest.mock.Mock()
    response.status_code = 200
    response.headers = {"Content-Encoding": "gzip"}
    return response


@pytest.mark.parametrize("request_method", ["get", "post", "delete"])
@unittest.mock.patch("time.sleep")
@unittest.mock.patch("requests.Session")
def test_timeout_is_forwarded_to_session(Session_class_mock, _sleep, request_method):
    """timeout=(connect_timeout, read_timeout) must be passed to every session verb."""
    session_mock = Session_class_mock.return_value.__enter__.return_value
    verb_mock = getattr(session_mock, request_method)
    verb_mock.return_value = _make_ok_response()

    with openml.config.overwrite_config_context(
        {"connect_timeout": 5.0, "read_timeout": 42.0, "connection_n_retries": 1}
    ):
        openml._api_calls._send_request(request_method, "http://example.com", {})

    verb_mock.assert_called_once()
    _call_kwargs = verb_mock.call_args[1]
    assert _call_kwargs["timeout"] == (5.0, 42.0)


@unittest.mock.patch("time.sleep")
@unittest.mock.patch("requests.Session")
def test_requests_timeout_is_retried_and_recovers(Session_class_mock, _sleep):
    """requests.Timeout on the first attempt must trigger a retry that succeeds."""
    session_mock = Session_class_mock.return_value.__enter__.return_value
    ok_response = _make_ok_response()
    session_mock.get.side_effect = [requests.exceptions.Timeout(), ok_response]

    with openml.config.overwrite_config_context({"connection_n_retries": 2}):
        result = openml._api_calls._send_request("get", "http://example.com", {})

    assert result is ok_response
    assert session_mock.get.call_count == 2
    _sleep.assert_called_once()  # exactly one inter-retry sleep


@unittest.mock.patch("time.sleep")
@unittest.mock.patch("requests.Session")
def test_requests_timeout_raised_after_all_retries_exhausted(Session_class_mock, _sleep):
    """requests.Timeout must propagate once every retry attempt is consumed."""
    session_mock = Session_class_mock.return_value.__enter__.return_value
    session_mock.get.side_effect = requests.exceptions.Timeout()

    with openml.config.overwrite_config_context({"connection_n_retries": 3}):
        with pytest.raises(requests.exceptions.Timeout):
            openml._api_calls._send_request("get", "http://example.com", {})

    assert session_mock.get.call_count == 3
    assert _sleep.call_count == 2  # sleep between retries, not after the last failure
