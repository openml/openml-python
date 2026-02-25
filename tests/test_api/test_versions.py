import pytest
from requests import Session, Response
from unittest.mock import patch
from openml._api import FallbackProxy, ResourceAPI, ResourceV1API, ResourceV2API, TaskAPI
from openml.enums import ResourceType
from openml.exceptions import OpenMLNotSupportedError
import openml


class DummyTaskAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.TASK


class DummyTaskV1API(ResourceV1API, TaskAPI):
    pass


class DummyTaskV2API(ResourceV2API, TaskAPI):
    pass


@pytest.fixture
def dummy_task_v1(http_client_v1, minio_client) -> DummyTaskV1API:
    return DummyTaskV1API(http=http_client_v1, minio=minio_client)


@pytest.fixture
def dummy_task_v2(http_client_v2, minio_client) -> DummyTaskV1API:
    return DummyTaskV2API(http=http_client_v2, minio=minio_client)


@pytest.fixture
def dummy_task_fallback(dummy_task_v1, dummy_task_v2) -> DummyTaskV1API:
    return FallbackProxy(dummy_task_v2, dummy_task_v1)


def _publish(resource):
    resource_name = resource.resource_type.value
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

        published_resource_id = resource.publish(
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


def _delete(resource):
    resource_name = resource.resource_type.value
    resource_id = 123

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:task_delete xmlns:oml="http://openml.org/openml">\n'
            f"  <oml:id>{resource_id}</oml:id>\n"
            f"</oml:task_delete>\n"
        ).encode("utf-8")

        resource.delete(resource_id)

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

def _tag(resource):
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

        tags = resource.tag(resource_id, resource_tag)

        assert resource_tag in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=(
                openml.config.server
                + resource.resource_type
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


def _untag(resource):
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

        tags = resource.untag(resource_id, resource_tag)

        assert resource_tag not in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=(
                openml.config.server
                + resource.resource_type
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



def test_v1_publish(dummy_task_v1, use_api_v1):
    _publish(dummy_task_v1)


def test_v1_delete(dummy_task_v1, use_api_v1):
    _delete(dummy_task_v1)


def test_v1_tag(dummy_task_v1, use_api_v1):
    _tag(dummy_task_v1)


def test_v1_untag(dummy_task_v1, use_api_v1):
    _untag(dummy_task_v1)


def test_v2_publish_not_supported(dummy_task_v2, use_api_v2):
    with pytest.raises(OpenMLNotSupportedError):
        _publish(dummy_task_v2)


def test_v2_delete_not_supported(dummy_task_v2, use_api_v2):
    with pytest.raises(OpenMLNotSupportedError):
        _delete(dummy_task_v2)


def test_v2_tag_not_supported(dummy_task_v2, use_api_v2):
    with pytest.raises(OpenMLNotSupportedError):
        _tag(dummy_task_v2)


def test_v2_untag_not_supported(dummy_task_v2, use_api_v2):
    with pytest.raises(OpenMLNotSupportedError):
        _untag(dummy_task_v2)


def test_fallback_publish(dummy_task_fallback, use_api_v1):
    _publish(dummy_task_fallback)


def test_fallback_delete(dummy_task_fallback, use_api_v1):
    _delete(dummy_task_fallback)


def test_fallback_tag(dummy_task_fallback, use_api_v1):
    _tag(dummy_task_fallback)


def test_fallback_untag(dummy_task_fallback, use_api_v1):
    _untag(dummy_task_fallback)
