import pytest
import pandas as pd
from requests import Session, Response
from unittest.mock import patch

import openml
from openml._api.resources.task import TaskV1API, TaskV2API
from openml._api.resources.base.fallback import FallbackProxy
from openml.exceptions import OpenMLNotSupportedError
from openml.tasks.task import TaskType


@pytest.fixture
def task_v1(http_client_v1, minio_client) -> TaskV1API:
    return TaskV1API(http=http_client_v1, minio=minio_client)


@pytest.fixture
def task_v2(http_client_v2, minio_client) -> TaskV2API:
    return TaskV2API(http=http_client_v2, minio=minio_client)


@pytest.fixture
def task_fallback(task_v1, task_v2) -> FallbackProxy:
    return FallbackProxy(task_v2, task_v1)


def _get_first_tid(task_api: TaskV1API, task_type: TaskType) -> int:
    """Helper to find an existing task ID for a given type using the V1 resource."""
    tasks = task_api.list(limit=1, offset=0, task_type=task_type)
    if tasks.empty:
        pytest.skip(f"No tasks of type {task_type} found on test server.")
    return int(tasks.iloc[0]["tid"])


@pytest.mark.uses_test_server()
def test_v1_list_tasks(task_v1):
    """Verify V1 list endpoint returns a populated DataFrame."""
    tasks_df = task_v1.list(limit=5, offset=0)
    assert isinstance(tasks_df, pd.DataFrame)
    assert not tasks_df.empty
    assert "tid" in tasks_df.columns


@pytest.mark.uses_test_server()
def test_v2_list_tasks(task_v2):
    """Verify V2 list endpoint raises NotSupported."""
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.list(limit=5, offset=0)

def test_v1_publish(task_v1):
    resource_name = task_v1.resource_type.value
    resource_files = {"description": "Resource Description File"}
    resource_id = 123

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:upload_task xmlns:oml="http://openml.org/openml">\n'
            f"\t<oml:id>{resource_id}</oml:id>\n"
            f"</oml:upload_task>\n"
        ).encode("utf-8")

        published_resource_id = task_v1.publish(
            resource_name,
            files=resource_files,
        )

        assert resource_id == published_resource_id

        mock_request.assert_called_once_with(
            method="POST",
            url=openml.config.server + resource_name,
            params={},
            data={"api_key": openml.config.apikey},
            headers=openml.config._HEADERS,
            files=resource_files,
        )


def test_v1_delete(task_v1):
    resource_name = task_v1.resource_type.value
    resource_id = 123

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:task_delete xmlns:oml="http://openml.org/openml">\n'
            f"  <oml:id>{resource_id}</oml:id>\n"
            f"</oml:task_delete>\n"
        ).encode("utf-8")

        task_v1.delete(resource_id)

        mock_request.assert_called_once_with(
            method="DELETE",
            url=(
                openml.config.server
                + resource_name
                + "/"
                + str(resource_id)
            ),
            params={"api_key": openml.config.apikey},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v1_tag(task_v1):
    resource_id = 123
    resource_tag = "TAG"

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:task_tag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{resource_id}</oml:id>"
            f"<oml:tag>{resource_tag}</oml:tag>"
            f"</oml:task_tag>"
        ).encode("utf-8")

        tags = task_v1.tag(resource_id, resource_tag)

        assert resource_tag in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=(
                openml.config.server
                + task_v1.resource_type.value
                + "/tag"
            ),
            params={},
            data={
                "api_key": openml.config.apikey,
                "task_id": resource_id,
                "tag": resource_tag,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v1_untag(task_v1):
    resource_id = 123
    resource_tag = "TAG"

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:task_untag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{resource_id}</oml:id>"
            f"</oml:task_untag>"
        ).encode("utf-8")

        tags = task_v1.untag(resource_id, resource_tag)

        assert resource_tag not in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=(
                openml.config.server
                + task_v1.resource_type.value
                + "/untag"
            ),
            params={},
            data={
                "api_key": openml.config.apikey,
                "task_id": resource_id,
                "tag": resource_tag,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v2_publish(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.publish(path=None, files=None)


def test_v2_delete(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.delete(resource_id=None)


def test_v2_tag(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.tag(resource_id=None, tag=None)


def test_v2_untag(task_v2):
    with pytest.raises(OpenMLNotSupportedError):
        task_v2.untag(resource_id=None, tag=None)
