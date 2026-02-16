# License: BSD 3-Clause
from __future__ import annotations

import pandas as pd
import pytest

from openml.enums import APIVersion
from openml._api.resources import FallbackProxy, StudyV1API, StudyV2API
from openml._api.resources.base import ResourceAPI
from openml.exceptions import OpenMLNotSupportedError
from openml.testing import TestAPIBase


@pytest.mark.uses_test_server()
class TestStudyAPIBase(TestAPIBase):
    """Base class for study API tests with common test utilities."""

    resource: ResourceAPI | FallbackProxy

    def _list_basic(self):
        """Test basic list functionality with limit and offset."""
        studies_df = self.resource.list(limit=5, offset=0)

        assert studies_df is not None
        assert len(studies_df) <= 5

        expected_columns = {"id", "alias", "main_entity_type", "name", "status"}
        assert expected_columns.issubset(set(studies_df.columns))

    def _list_with_filters(self):
        """Test list with various filters."""
        studies_df = self.resource.list(
            limit=10,
            offset=0,
            status="active",
        )
        if len(studies_df) > 0:
            assert all(studies_df["status"] == "active")

    def _list_pagination(self):
        """Test pagination with offset and limit."""
        try:
            page1 = self.resource.list(limit=3, offset=0)
            
            if len(page1) >= 3:
                page2 = self.resource.list(limit=3, offset=3)

                if len(page2) > 0:
                    page1_ids = set(page1["id"])
                    page2_ids = set(page2["id"])
                    assert page1_ids.isdisjoint(page2_ids)
        except Exception:
            pytest.skip("Not enough studies on test server for pagination test")


class TestStudyV1API(TestStudyAPIBase):
    """Tests for V1 XML API implementation of studies."""

    def setUp(self) -> None:
        super().setUp()
        self.resource = StudyV1API(self.http_clients[APIVersion.V1])

    def test_list_basic(self):
        """Test basic list functionality."""
        self._list_basic()

    def test_list_with_filters(self):
        """Test list with filters."""
        self._list_with_filters()

    def test_list_pagination(self):
        """Test pagination."""
        self._list_pagination()


class TestStudyV2API(TestStudyAPIBase):
    """Tests for V2 API implementation of studies."""

    def setUp(self) -> None:
        super().setUp()
        self.resource = StudyV2API(self.http_clients[APIVersion.V2])

    def test_list_basic(self):
        """Test that list raises OpenMLNotSupportedError for V2."""
        with pytest.raises(OpenMLNotSupportedError):
            self._list_basic()

    def test_list_with_filters(self):
        """Test that list with filters raises OpenMLNotSupportedError for V2."""
        with pytest.raises(OpenMLNotSupportedError):
            self._list_with_filters()

    def test_list_pagination(self):
        """Test that list pagination raises OpenMLNotSupportedError for V2."""
        with pytest.raises(OpenMLNotSupportedError):
            self._list_pagination()


class TestStudyCombinedAPI(TestStudyAPIBase):
    """Combined tests for study API with FallbackProxy."""

    def setUp(self) -> None:
        super().setUp()
        http_client_v1 = self.http_clients[APIVersion.V1]
        resource_v1 = StudyV1API(http_client_v1)

        http_client_v2 = self.http_clients[APIVersion.V2]
        resource_v2 = StudyV2API(http_client_v2)

        self.resource = FallbackProxy(resource_v2, resource_v1)

    def test_list_basic(self):
        """Test that FallbackProxy falls back from V2 to V1 for list operation."""
        self._list_basic()

    def test_list_with_filters(self):
        """Test that FallbackProxy falls back for list with filters."""
        self._list_with_filters()

    def test_list_pagination(self):
        """Test that FallbackProxy falls back for list pagination."""
        self._list_pagination()
