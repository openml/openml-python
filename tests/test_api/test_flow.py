# License: BSD 3-Clause
from __future__ import annotations

from unittest.mock import patch

import pytest
from requests import Response, Session

import openml
from openml._api import FlowV1API, FlowV2API
from openml.exceptions import OpenMLNotSupportedError, OpenMLServerException
from openml.flows.flow import OpenMLFlow


@pytest.fixture
def flow_v1(http_client_v1, minio_client) -> FlowV1API:
    return FlowV1API(http=http_client_v1, minio=minio_client)


@pytest.fixture
def flow_v2(http_client_v2, minio_client) -> FlowV2API:
    from openml.enums import APIVersion

    if openml.config.servers[APIVersion.V2]["server"] is None:
        pytest.skip("V2 server is not configured")
    return FlowV2API(http=http_client_v2, minio=minio_client)


def _assert_flow_shape(flow: OpenMLFlow) -> None:
    assert isinstance(flow, OpenMLFlow)
    assert isinstance(flow.flow_id, int)
    assert flow.flow_id > 0
    assert isinstance(flow.name, str)
    assert len(flow.name) > 0


# ---------------------------------------------------------------------------
# V1 tests
# ---------------------------------------------------------------------------


@pytest.mark.test_server()
def test_flow_v1_get(flow_v1):
    flow = flow_v1.get(flow_id=1)
    _assert_flow_shape(flow)


@pytest.mark.test_server()
def test_flow_v1_list(flow_v1):
    limit = 5
    flows_df = flow_v1.list(limit=limit)

    assert len(flows_df) == limit
    assert "id" in flows_df.columns
    assert "name" in flows_df.columns
    assert "version" in flows_df.columns
    assert "external_version" in flows_df.columns
    assert "full_name" in flows_df.columns
    assert "uploader" in flows_df.columns


@pytest.mark.test_server()
def test_flow_v1_list_with_offset(flow_v1):
    limit = 5
    flows_df = flow_v1.list(limit=limit, offset=10)

    assert len(flows_df) == limit


def test_flow_v1_exists_input_validation(flow_v1):
    with pytest.raises(ValueError, match="Argument 'name' should be a non-empty string"):
        flow_v1.exists(name="", external_version="1")

    with pytest.raises(ValueError, match="Argument 'version' should be a non-empty string"):
        flow_v1.exists(name="sklearn.tree.DecisionTreeClassifier", external_version="")


def test_flow_v1_exists_mocked_success(flow_v1):
    flow_name = "sklearn.tree.DecisionTreeClassifier"
    external_version = "1"

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:flow_exists xmlns:oml="http://openml.org/openml">\n'
            "  <oml:id>123</oml:id>\n"
            "</oml:flow_exists>\n"
        ).encode("utf-8")

        result = flow_v1.exists(name=flow_name, external_version=external_version)

        assert result == 123
        mock_request.assert_called_once()
        _, kwargs = mock_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == openml.config.server + "flow/exists"
        assert kwargs["params"] == {}
        assert kwargs["headers"] == openml.config._HEADERS
        assert kwargs["files"] is None
        assert kwargs["data"]["name"] == flow_name
        assert kwargs["data"]["external_version"] == external_version


def test_flow_v1_exists_mocked_server_error(flow_v1):
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:error xmlns:oml="http://openml.org/openml">\n'
            "  <oml:code>104</oml:code>\n"
            "  <oml:message>Server error</oml:message>\n"
            "</oml:error>\n"
        ).encode("utf-8")

        with pytest.raises(OpenMLServerException, match="Server error"):
            flow_v1.exists(name="foo", external_version="1")


def test_flow_v1_publish_mocked(flow_v1, test_apikey_v1):
    files = {"description": "<flow/>"}

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:upload_flow xmlns:oml="http://openml.org/openml">\n'
            "  <oml:id>321</oml:id>\n"
            "</oml:upload_flow>\n"
        ).encode("utf-8")

        result = flow_v1.publish(path="flow", files=files)

        assert result == 321
        mock_request.assert_called_once_with(
            method="POST",
            url=openml.config.server + "flow",
            params={},
            data={"api_key": test_apikey_v1},
            headers=openml.config._HEADERS,
            files=files,
        )


def test_flow_v1_delete_mocked(flow_v1, test_apikey_v1):
    flow_id = 123

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:flow_delete xmlns:oml="http://openml.org/openml">\n'
            "  <oml:id>123</oml:id>\n"
            "</oml:flow_delete>\n"
        ).encode("utf-8")

        result = flow_v1.delete(flow_id)

        assert result is True
        mock_request.assert_called_once_with(
            method="DELETE",
            url=openml.config.server + f"flow/{flow_id}",
            params={"api_key": test_apikey_v1},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )


def test_flow_v1_tag_mocked(flow_v1, test_apikey_v1):
    """Test V1 tagging a flow."""
    flow_id = 42
    tag_name = "my-test-tag"

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:flow_tag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{flow_id}</oml:id>"
            f"<oml:tag>{tag_name}</oml:tag>"
            "</oml:flow_tag>"
        ).encode("utf-8")

        tags = flow_v1.tag(flow_id, tag_name)

        assert tag_name in tags
        mock_request.assert_called_once_with(
            method="POST",
            url=openml.config.server + "flow/tag",
            params={},
            data={
                "api_key": test_apikey_v1,
                "flow_id": flow_id,
                "tag": tag_name,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


def test_flow_v1_untag_mocked(flow_v1, test_apikey_v1):
    """Test V1 untagging a flow."""
    flow_id = 42
    tag_name = "my-test-tag"

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            '<oml:flow_untag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{flow_id}</oml:id>"
            "</oml:flow_untag>"
        ).encode("utf-8")

        tags = flow_v1.untag(flow_id, tag_name)

        assert tag_name not in tags
        mock_request.assert_called_once_with(
            method="POST",
            url=openml.config.server + "flow/untag",
            params={},
            data={
                "api_key": test_apikey_v1,
                "flow_id": flow_id,
                "tag": tag_name,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


# ---------------------------------------------------------------------------
# V2 tests
# ---------------------------------------------------------------------------


@pytest.mark.test_server()
def test_flow_v2_get(flow_v2):
    flow = flow_v2.get(flow_id=1)
    _assert_flow_shape(flow)


def test_flow_v2_exists_nonexistent(flow_v2):
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = b"{}"
        mock_request.return_value.json = lambda: {"flow_id": False}

        result = flow_v2.exists(
            name="NonExistentFlowName123456789",
            external_version="0.0.0.nonexistent",
        )

    assert result is False


def test_flow_v2_list_not_supported(flow_v2):
    with pytest.raises(
        OpenMLNotSupportedError,
        match="FlowV2API: v2 API does not support `list` for resource `flow`",
    ):
        flow_v2.list(limit=10)


def test_flow_v2_publish_not_supported(flow_v2):
    with pytest.raises(
        OpenMLNotSupportedError,
        match="FlowV2API: v2 API does not support `publish` for resource `flow`",
    ):
        flow_v2.publish(path="flow", files={"description": "<flow/>"})


@pytest.mark.test_server()
def test_flow_v1_v2_get_output_match(flow_v1, flow_v2):
    flow_from_v1 = flow_v1.get(flow_id=1)
    flow_from_v2 = flow_v2.get(flow_id=1)

    assert flow_from_v1.flow_id == flow_from_v2.flow_id
    assert flow_from_v1.name == flow_from_v2.name
    assert flow_from_v1.version == flow_from_v2.version
    assert flow_from_v1.external_version == flow_from_v2.external_version
    assert flow_from_v1.description == flow_from_v2.description
