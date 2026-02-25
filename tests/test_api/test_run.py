# License: BSD 3-Clause
from __future__ import annotations

from unittest.mock import patch

import pytest
from requests import Response, Session

import openml
from openml._api import RunV1API, RunV2API
from openml.exceptions import OpenMLNotSupportedError
from openml.runs.run import OpenMLRun


@pytest.fixture
def run_v1(http_client_v1, minio_client) -> RunV1API:
    return RunV1API(http=http_client_v1, minio=minio_client)


@pytest.fixture
def run_v2(http_client_v2, minio_client) -> RunV2API:
    return RunV2API(http=http_client_v2, minio=minio_client)


def _assert_run_shape(run: OpenMLRun) -> None:
    assert isinstance(run, OpenMLRun)
    assert isinstance(run.run_id, int)
    assert run.run_id > 0
    assert isinstance(run.task_id, int)


@pytest.mark.test_server()
def test_run_v1_get(run_v1):
    run = run_v1.get(run_id=1)
    _assert_run_shape(run)


@pytest.mark.test_server()
def test_run_v1_list(run_v1):
    limit = 5
    runs_df = run_v1.list(limit=limit, offset=0)

    assert len(runs_df) == limit
    assert "run_id" in runs_df.columns
    assert "task_id" in runs_df.columns
    assert "setup_id" in runs_df.columns
    assert "flow_id" in runs_df.columns


def test_run_v1_publish_mocked(run_v1, use_api_v1, test_api_key):
    files = {"description": "<run/>"}

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:upload_run xmlns:oml="http://openml.org/openml">\n'
            "  <oml:id>456</oml:id>\n"
            "</oml:upload_run>\n"
        ).encode("utf-8")

        result = run_v1.publish(path="run", files=files)

        assert result == 456
        mock_request.assert_called_once_with(
            method="POST",
            url=openml.config.server + "run",
            params={},
            data={"api_key": test_api_key},
            headers=openml.config._HEADERS,
            files=files,
        )


def test_run_v1_delete_mocked(run_v1, use_api_v1, test_api_key):
    run_id = 456

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:run_delete xmlns:oml="http://openml.org/openml">\n'
            "  <oml:id>456</oml:id>\n"
            "</oml:run_delete>\n"
        ).encode("utf-8")

        result = run_v1.delete(run_id)

        assert result is True
        mock_request.assert_called_once_with(
            method="DELETE",
            url=openml.config.server + f"run/{run_id}",
            params={"api_key": test_api_key},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )


def test_run_v2_get_not_supported(run_v2):
    with pytest.raises(
        OpenMLNotSupportedError,
        match="RunV2API: v2 API does not support `get` for resource `run`",
    ):
        run_v2.get(run_id=1)


def test_run_v2_list_not_supported(run_v2):
    with pytest.raises(
        OpenMLNotSupportedError,
        match="RunV2API: v2 API does not support `list` for resource `run`",
    ):
        run_v2.list(limit=5, offset=0)


def test_run_v2_publish_not_supported(run_v2):
    with pytest.raises(
        OpenMLNotSupportedError,
        match="RunV2API: v2 API does not support `publish` for resource `run`",
    ):
        run_v2.publish(path="run", files={"description": "<run/>"})


@pytest.mark.test_server()
def test_run_v1_v2_contracts(run_v1, run_v2):
    run_from_v1 = run_v1.get(run_id=1)
    _assert_run_shape(run_from_v1)

    with pytest.raises(OpenMLNotSupportedError, match="does not support `get`"):
        run_v2.get(run_id=1)

    with pytest.raises(OpenMLNotSupportedError, match="does not support `list`"):
        run_v2.list(limit=5, offset=0)

    with pytest.raises(OpenMLNotSupportedError, match="does not support `publish`"):
        run_v2.publish(path="run", files={"description": "<run/>"})
