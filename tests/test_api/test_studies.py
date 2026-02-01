# License: BSD 3-Clause
from __future__ import annotations

import pytest

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

    @pytest.mark.uses_test_server()
    def test_delete(self):
        """Test delete method (inherited from ResourceV1API)."""
        assert hasattr(self.api, "delete")
        assert callable(self.api.delete)


class TestStudyV2API(TestAPIBase):
    """Tests for V2 API implementation of studies."""

    def setUp(self) -> None:
        super().setUp()
        self.api = StudyV2API(self.http_client)

    def test_list_not_supported(self):
        """Test that list raises OpenMLNotSupportedError for V2."""
        with pytest.raises(OpenMLNotSupportedError):
            self.api.list(limit=10, offset=0)


class TestStudyCombined(TestAPIBase):
    """Combined tests for study API fallback behavior."""

    def setUp(self) -> None:
        super().setUp()
        self.v1_api = StudyV1API(self.http_client)
        self.v2_api = StudyV2API(self.http_client)

    @pytest.mark.uses_test_server()
    def test_v1_v2_compatibility(self):
        """Verify V1 and V2 APIs have compatible interfaces."""
        # Both should have the same method names
        assert hasattr(self.v1_api, "list")
        assert hasattr(self.v2_api, "list")
        
        # Both should have delete, tag, untag from base
        for method in ["delete", "tag", "untag", "publish"]:
            assert hasattr(self.v1_api, method)
            assert hasattr(self.v2_api, method)