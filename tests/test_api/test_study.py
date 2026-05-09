# License: BSD 3-Clause
from __future__ import annotations

import pytest
from requests import Session, Response
from unittest.mock import patch
import pandas as pd

from openml._api.resources import StudyV1API, StudyV2API
from openml.exceptions import OpenMLNotSupportedError
import openml


@pytest.fixture
def study_v1(http_client_v1, minio_client) -> StudyV1API:
    """Fixture for V1 Study API instance."""
    return StudyV1API(http=http_client_v1, minio=minio_client)


@pytest.fixture
def study_v2(http_client_v2, minio_client) -> StudyV2API:
    """Fixture for V2 Study API instance."""
    return StudyV2API(http=http_client_v2, minio=minio_client)


def test_v1_list(study_v1, test_server_v1, test_apikey_v1):
    """Test V1 list basic functionality with limit and offset."""
    # Mock response with study list
    mock_response = """<?xml version="1.0" encoding="UTF-8"?>
    <oml:study_list xmlns:oml="http://openml.org/openml">
        <oml:study>
            <oml:id>1</oml:id>
            <oml:alias>test-study-1</oml:alias>
            <oml:main_entity_type>task</oml:main_entity_type>
            <oml:name>Test Study 1</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
        <oml:study>
            <oml:id>2</oml:id>
            <oml:alias>test-study-2</oml:alias>
            <oml:main_entity_type>run</oml:main_entity_type>
            <oml:name>Test Study 2</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
    </oml:study_list>
    """
    limit = 5
    offset = 0

    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = mock_response.encode("utf-8")

        studies_df = study_v1.list(limit=limit, offset=offset)

        assert studies_df is not None
        assert isinstance(studies_df, pd.DataFrame)
        assert len(studies_df) == 2

        expected_columns = {"id", "alias", "main_entity_type", "name", "status"}
        assert expected_columns.issubset(set(studies_df.columns))

        mock_request.assert_called_once_with(
            method="GET",
            url=test_server_v1 + f"study/list/limit/{limit}/offset/{offset}",
            params={},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v1_publish(study_v1, test_server_v1, test_apikey_v1):
    """Test V1 publish a new study."""
    study_id = 999
    study_files = {"description": "Test Study Description"}
    
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:upload_study xmlns:oml="http://openml.org/openml">\n'
            f"\t<oml:id>{study_id}</oml:id>\n"
            f"</oml:upload_study>\n"
        ).encode("utf-8")

        published_id = study_v1.publish("study", files=study_files)

        assert published_id == study_id

        mock_request.assert_called_once_with(
            method="POST",
            url=test_server_v1 + "study",
            params={},
            data={"api_key": test_apikey_v1},
            headers=openml.config._HEADERS,
            files=study_files,
        )


def test_v1_delete(study_v1, test_server_v1, test_apikey_v1):
    """Test V1 delete a study."""
    study_id = 100
    
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:study_delete xmlns:oml="http://openml.org/openml">\n'
            f"  <oml:id>{study_id}</oml:id>\n"
            f"</oml:study_delete>\n"
        ).encode("utf-8")

        result = study_v1.delete(study_id)

        assert result

        mock_request.assert_called_once_with(
            method="DELETE",
            url=test_server_v1 + "study/" + str(study_id),
            params={"api_key": test_apikey_v1},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v1_tag(study_v1, test_server_v1, test_apikey_v1):
    """Test V1 tag a study."""
    study_id = 100
    tag_name = "important-tag"
    
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:study_tag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{study_id}</oml:id>"
            f"<oml:tag>{tag_name}</oml:tag>"
            f"</oml:study_tag>"
        ).encode("utf-8")

        tags = study_v1.tag(study_id, tag_name)

        assert tag_name in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=test_server_v1 + "study/tag",
            params={},
            data={
                "api_key": test_apikey_v1,
                "study_id": study_id,
                "tag": tag_name,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v1_untag(study_v1, test_server_v1, test_apikey_v1):
    """Test V1 untag a study."""
    study_id = 100
    tag_name = "important-tag"
    
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = (
            f'<oml:study_untag xmlns:oml="http://openml.org/openml">'
            f"<oml:id>{study_id}</oml:id>"
            f"</oml:study_untag>"
        ).encode("utf-8")

        tags = study_v1.untag(study_id, tag_name)

        assert tag_name not in tags

        mock_request.assert_called_once_with(
            method="POST",
            url=test_server_v1 + "study/untag",
            params={},
            data={
                "api_key": test_apikey_v1,
                "study_id": study_id,
                "tag": tag_name,
            },
            headers=openml.config._HEADERS,
            files=None,
        )


def test_v2_list(study_v2):
    """Test that V2 list raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.list(limit=5, offset=0)


def test_v2_publish(study_v2):
    """Test that V2 publish raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.publish(path="study", files=None)


def test_v2_delete(study_v2):
    """Test that V2 delete raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.delete(resource_id=100)


def test_v2_tag(study_v2):
    """Test that V2 tag raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.tag(resource_id=100, tag="test-tag")


def test_v2_untag(study_v2):
    """Test that V2 untag raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.untag(resource_id=100, tag="test-tag")
