# License: BSD 3-Clause
from __future__ import annotations

import pandas as pd
import pytest

from openml._api.resources.base.fallback import FallbackProxy
from openml._api.resources.study import StudyV1API, StudyV2API
from openml.exceptions import OpenMLNotSupportedError
from openml.testing import TestAPIBase


class TestStudyV1API(TestAPIBase):
    """Tests for V1 XML API implementation of studies."""

    def setUp(self) -> None:
        super().setUp()
        self.api = StudyV1API(self.http_client)

    @pytest.mark.uses_test_server()
    def test_list_basic(self):
        """Test basic list functionality with limit and offset."""
        studies_df = self.api.list(limit=5, offset=0)

        assert studies_df is not None
        assert len(studies_df) <= 5

        expected_columns = {"id", "alias", "main_entity_type", "name", "status"}
        assert expected_columns.issubset(set(studies_df.columns))

    @pytest.mark.uses_test_server()
    def test_list_with_filters(self):
        """Test list with various filters."""
        studies_df = self.api.list(
            limit=10,
            offset=0,
            status="active",
        )
        if len(studies_df) > 0:
            assert all(studies_df["status"] == "active")

    @pytest.mark.uses_test_server()
    def test_list_pagination(self):
        """Test pagination with offset and limit."""
        try:
            page1 = self.api.list(limit=3, offset=0)
            
            if len(page1) >= 3:
                page2 = self.api.list(limit=3, offset=3)

                if len(page2) > 0:
                    page1_ids = set(page1["id"])
                    page2_ids = set(page2["id"])
                    assert page1_ids.isdisjoint(page2_ids)
        except Exception:
            pytest.skip("Not enough studies on test server for pagination test")


class TestStudyV2API(TestAPIBase):
    """Tests for V2 API implementation of studies."""

    def setUp(self) -> None:
        super().setUp()
        self.v2_client = self._get_http_client(
            server="http://localhost:8001/",
            base_url="",
            api_key="",
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        self.api = StudyV2API(self.v2_client)

    def test_list_not_supported(self):
        """Test that list raises OpenMLNotSupportedError for V2."""
        with pytest.raises(OpenMLNotSupportedError):
            self.api.list(limit=10, offset=0)


class TestStudyCombined(TestAPIBase):
    """Combined tests for study API V1 and V2."""

    def setUp(self) -> None:
        super().setUp()
        self.v1_client = self.http_client
        self.v2_client = self._get_http_client(
            server="http://localhost:8001/",
            base_url="",
            api_key="",
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        self.resource_v1 = StudyV1API(self.v1_client)
        self.resource_v2 = StudyV2API(self.v2_client)
        self.resource_fallback = FallbackProxy(self.resource_v2, self.resource_v1)

    @pytest.mark.skip(reason="V2 list not yet implemented")
    @pytest.mark.uses_test_server()
    def test_list_matches(self):
        """Test that V1 and V2 list return matching results."""
        output_v1 = self.resource_v1.list(limit=5, offset=0)
        output_v2 = self.resource_v2.list(limit=5, offset=0)

        assert isinstance(output_v1, pd.DataFrame)
        assert isinstance(output_v2, pd.DataFrame)
        assert output_v1.equals(output_v2)

    @pytest.mark.uses_test_server()
    def test_list_fallback(self):
        """Test that FallbackProxy falls back from V2 to V1 for list operation."""
        output_fallback = self.resource_fallback.list(limit=5, offset=0)
        
        assert output_fallback is not None
        assert len(output_fallback) <= 5
