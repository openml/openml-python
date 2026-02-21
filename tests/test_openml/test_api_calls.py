from __future__ import annotations

import unittest.mock
from pathlib import Path
from typing import NamedTuple, Iterable, Iterator
from unittest import mock

import minio
import pytest

import openml
import openml.testing
from openml._api_calls import _download_minio_bucket, API_TOKEN_HELP_LINK


class TestConfig(openml.testing.TestBase):

    @unittest.mock.patch("requests.Session")
    def test_too_long_uri(self, Session_class_mock):
        response_mock = unittest.mock.Mock()
        response_mock.status_code = 414
        response_mock.text = "URI too long!"
        response_mock.headers = {}

        session_instance = Session_class_mock.return_value.__enter__.return_value
        session_instance.get.return_value = response_mock

        with pytest.raises(openml.exceptions.OpenMLServerError, match="URI too long!"):
            openml.datasets.list_datasets(data_id=list(range(10000)))

        session_instance.get.assert_called_once()

    @unittest.mock.patch("time.sleep")
    @unittest.mock.patch("requests.Session")
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

        session_instance = Session_class_mock.return_value.__enter__.return_value
        session_instance.get.return_value = response_mock

        with pytest.raises(openml.exceptions.OpenMLServerException, match="/abc returned code 107"):
            openml._api_calls._send_request("get", "/abc", {})

        assert session_instance.get.call_count == 20


class FakeObject(NamedTuple):
    object_name: str
    etag: str


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

    file_path = tmp_path / some_filename

    _download_minio_bucket(source=some_url, destination=tmp_path)
    mtime_first = file_path.stat().st_mtime

    _download_minio_bucket(source=some_url, destination=tmp_path)
    mtime_second = file_path.stat().st_mtime

    assert mtime_first == mtime_second


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


@unittest.mock.patch("requests.Session")
@pytest.mark.parametrize(
    "endpoint, method",
    [
        ("flow/exists", "post"),
        ("dataset", "post"),
        ("dataset/42", "delete"),
        ("flow/42", "delete"),
        ("run/42", "delete"),
        ("task/42", "delete"),
    ],
)
def test_authentication_endpoints_requiring_api_key_show_relevant_help_link(
    Session_class_mock,
    endpoint: str,
    method: str,
) -> None:
    response_mock = unittest.mock.Mock()
    response_mock.status_code = 200
    response_mock.headers = {}
    response_mock.text = (
        "<oml:error>"
        "<oml:code>401</oml:code>"
        "<oml:message>Authentication required</oml:message>"
        "</oml:error>"
    )

    session_instance = Session_class_mock.return_value.__enter__.return_value
    session_instance.request.return_value = response_mock

    with openml.config.overwrite_config_context({"apikey": None}):
        with pytest.raises(
            openml.exceptions.OpenMLNotAuthorizedError,
            match=API_TOKEN_HELP_LINK,
        ):
            openml._api_calls._perform_api_call(
                call=endpoint,
                request_method=method,
                data=None,
            )

    session_instance.request.assert_called()
    