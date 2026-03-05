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
def study_v1(http_client_v1) -> StudyV1API:
    """Fixture for V1 Study API instance."""
    return StudyV1API(http=http_client_v1)


@pytest.fixture
def study_v2(http_client_v2) -> StudyV2API:
    """Fixture for V2 Study API instance."""
    return StudyV2API(http=http_client_v2)

def test_v1_list_basic(study_v1, test_server_v1, test_apikey_v1):
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
    
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = mock_response.encode("utf-8")

        studies_df = study_v1.list(limit=5, offset=0)

        assert studies_df is not None
        assert isinstance(studies_df, pd.DataFrame)
        assert len(studies_df) == 2

        expected_columns = {"id", "alias", "main_entity_type", "name", "status"}
        assert expected_columns.issubset(set(studies_df.columns))


def test_v1_list_with_status_filter(study_v1, test_server_v1, test_apikey_v1):
    """Test V1 list with status filter."""
    mock_response = """<?xml version="1.0" encoding="UTF-8"?>
    <oml:study_list xmlns:oml="http://openml.org/openml">
        <oml:study>
            <oml:id>1</oml:id>
            <oml:alias>active-study</oml:alias>
            <oml:main_entity_type>task</oml:main_entity_type>
            <oml:name>Active Study</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
    </oml:study_list>
    """
    
    with patch.object(Session, "request") as mock_request:
        mock_request.return_value = Response()
        mock_request.return_value.status_code = 200
        mock_request.return_value._content = mock_response.encode("utf-8")

        studies_df = study_v1.list(limit=10, offset=0, status="active")

        assert studies_df is not None
        assert all(studies_df["status"] == "active")

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert "/status/active" in call_args.kwargs.get("url", "")


def test_v1_list_pagination(study_v1, test_server_v1, test_apikey_v1):
    """Test V1 list pagination with offset and limit."""
    page1_response = """<?xml version="1.0" encoding="UTF-8"?>
    <oml:study_list xmlns:oml="http://openml.org/openml">
        <oml:study>
            <oml:id>1</oml:id>
            <oml:alias>study-1</oml:alias>
            <oml:main_entity_type>task</oml:main_entity_type>
            <oml:name>Study 1</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
        <oml:study>
            <oml:id>2</oml:id>
            <oml:alias>study-2</oml:alias>
            <oml:main_entity_type>task</oml:main_entity_type>
            <oml:name>Study 2</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
        <oml:study>
            <oml:id>3</oml:id>
            <oml:alias>study-3</oml:alias>
            <oml:main_entity_type>task</oml:main_entity_type>
            <oml:name>Study 3</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
    </oml:study_list>
    """
    
    page2_response = """<?xml version="1.0" encoding="UTF-8"?>
    <oml:study_list xmlns:oml="http://openml.org/openml">
        <oml:study>
            <oml:id>4</oml:id>
            <oml:alias>study-4</oml:alias>
            <oml:main_entity_type>run</oml:main_entity_type>
            <oml:name>Study 4</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
        <oml:study>
            <oml:id>5</oml:id>
            <oml:alias>study-5</oml:alias>
            <oml:main_entity_type>run</oml:main_entity_type>
            <oml:name>Study 5</oml:name>
            <oml:status>active</oml:status>
        </oml:study>
    </oml:study_list>
    """
    
    with patch.object(Session, "request") as mock_request:
        page1_response_obj = Response()
        page1_response_obj.status_code = 200
        page1_response_obj._content = page1_response.encode("utf-8")
        
        page2_response_obj = Response()
        page2_response_obj.status_code = 200
        page2_response_obj._content = page2_response.encode("utf-8")
        
        mock_request.side_effect = [page1_response_obj, page2_response_obj]

        page1 = study_v1.list(limit=3, offset=0)
        page2 = study_v1.list(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 2
        
        page1_ids = set(page1["id"])
        page2_ids = set(page2["id"])
        assert page1_ids.isdisjoint(page2_ids)


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

        assert result is True

        mock_request.assert_called_once_with(
            method="DELETE",
            url=test_server_v1 + "study/" + str(study_id),
            params={"api_key": test_apikey_v1},
            data={},
            headers=openml.config._HEADERS,
            files=None,
        )

def test_v2_list_not_supported(study_v2):
    """Test that V2 list raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.list(limit=5, offset=0)


def test_v2_publish_not_supported(study_v2):
    """Test that V2 publish raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.publish(path="study", files=None)


def test_v2_delete_not_supported(study_v2):
    """Test that V2 delete raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.delete(resource_id=100)


def test_v2_tag_not_supported(study_v2):
    """Test that V2 tag raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.tag(resource_id=100, tag="test-tag")


def test_v2_untag_not_supported(study_v2):
    """Test that V2 untag raises OpenMLNotSupportedError."""
    with pytest.raises(OpenMLNotSupportedError):
        study_v2.untag(resource_id=100, tag="test-tag")

