# License: BSD 3-Clause
"""Tests for Flow V1 → V2 API Migration."""
from __future__ import annotations

import pytest

from openml._api.resources import FallbackProxy, FlowsV1, FlowsV2
from openml.flows.flow import OpenMLFlow
from openml.testing import TestAPIBase


class TestFlowsV1(TestAPIBase):
    """Test FlowsV1 resource implementation."""

    def setUp(self):
        super().setUp()
        self.resource = FlowsV1(self.http_client)

    @pytest.mark.uses_test_server()
    def test_get(self):
        """Test getting a flow from the V1 API."""
        flow = self.resource.get(flow_id=1)
        
        assert isinstance(flow, OpenMLFlow)
        assert flow.flow_id == 1
        assert isinstance(flow.name, str)
        assert len(flow.name) > 0

    @pytest.mark.uses_test_server()
    def test_exists(self):
        """Test checking if a flow exists using V1 API."""
        flow = self.resource.get(flow_id=1)
        
        result = self.resource.exists(
            name=flow.name,
            external_version=flow.external_version
        )
        
        assert isinstance(result, int)
        assert result > 0
        assert result == flow.flow_id

    @pytest.mark.uses_test_server()
    def test_exists_nonexistent(self):
        """Test checking if a non-existent flow exists using V1 API."""
        result = self.resource.exists(
            name="NonExistentFlowName123456789",
            external_version="0.0.0.nonexistent"
        )
        
        assert result is False

    @pytest.mark.uses_test_server()
    def test_list(self):
        """Test listing flows from the V1 API."""
        flows_df = self.resource.list(limit=10)
        
        assert len(flows_df) > 0
        assert len(flows_df) <= 10
        assert "id" in flows_df.columns
        assert "name" in flows_df.columns
        assert "version" in flows_df.columns
        assert "external_version" in flows_df.columns
        assert "full_name" in flows_df.columns
        assert "uploader" in flows_df.columns

    @pytest.mark.uses_test_server()
    def test_list_with_offset(self):
        """Test listing flows with offset from the V1 API."""
        flows_df = self.resource.list(limit=5, offset=10)
        
        assert len(flows_df) > 0
        assert len(flows_df) <= 5

    @pytest.mark.uses_test_server()
    def test_list_with_tag_limit_offset(self):
        """Test listing flows with filters from the V1 API."""
        flows_df = self.resource.list(tag="weka", limit=5 , offset=0 , uploader=16)
        
        assert hasattr(flows_df, 'columns')
        if len(flows_df) > 0:
            assert "id" in flows_df.columns
    
    @pytest.mark.uses_test_server()
    def test_publish(self):
        """Test publishing a sklearn flow using V1 API."""
        from openml_sklearn.extension import SklearnExtension
        from sklearn.tree import ExtraTreeRegressor
        clf = ExtraTreeRegressor()
        extension = SklearnExtension()
        dt_flow = extension.model_to_flow(clf)
        published_flow = self.resource.publish(dt_flow)
        assert isinstance(published_flow, OpenMLFlow)
        assert getattr(published_flow, "id", None) is not None
    
    @pytest.mark.uses_test_server()
    def test_delete(self):
        """Test deleting a flow using V1 API."""
        from openml_sklearn.extension import SklearnExtension
        from sklearn.tree import ExtraTreeRegressor
        clf = ExtraTreeRegressor()
        extension = SklearnExtension()
        dt_flow = extension.model_to_flow(clf)
        flow_id = self.resource.exists(
            name=dt_flow.name,
            external_version=dt_flow.external_version
        )
        result = self.resource.delete(flow_id)
        assert result is True
        exists = self.resource.exists(
            name=dt_flow.name,
            external_version=dt_flow.external_version
        )
        assert exists is False



class TestFlowsV2(TestAPIBase):
    """Test FlowsV2 resource implementation."""

    def setUp(self):
        super().setUp()
        self.v2_http_client = self._get_http_client(
            server="http://127.0.0.1:8001/",
            base_url="", 
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        self.resource = FlowsV2(self.v2_http_client)

    # @pytest.mark.skip(reason="V2 API not yet deployed on test server")
    @pytest.mark.uses_test_server()
    def test_get(self):
        """Test getting a flow from the V2 API."""
        flow = self.resource.get(flow_id=1)
        
        assert isinstance(flow, OpenMLFlow)
        assert flow.flow_id == 1
        assert isinstance(flow.name, str)
        assert len(flow.name) > 0

    # @pytest.mark.skip(reason="V2 API not yet deployed on test server")
    @pytest.mark.uses_test_server()
    def test_exists(self):
        """Test checking if a flow exists using V2 API."""
        flow = self.resource.get(flow_id=1)
        
        result = self.resource.exists(
            name=flow.name,
            external_version=flow.external_version
        )
        
        # V2 may return int or bool
        assert result is not False
        if isinstance(result, int):
            assert result > 0

    # @pytest.mark.skip(reason="V2 API not yet deployed on test server")
    @pytest.mark.uses_test_server()
    def test_exists_nonexistent(self):
        """Test checking if a non-existent flow exists using V2 API."""
        result = self.resource.exists(
            name="NonExistentFlowName123456789",
            external_version="0.0.0.nonexistent"
        )
        
        assert result is False

    def test_list_not_implemented(self):
        """Test that list raises NotImplementedError for V2."""
        with pytest.raises(NotImplementedError):
            self.resource.list(limit=10)

    def test_publish_not_implemented(self):
        """Test that publish raises NotImplementedError for V2."""
        from collections import OrderedDict
        
        with pytest.raises(NotImplementedError):
            flow = OpenMLFlow(
                name="test",
                description="test",
                model=None,
                components=OrderedDict(),
                parameters=OrderedDict(),
                parameters_meta_info=OrderedDict(),
                external_version="1.0",
                tags=[],
                language="English",
                dependencies=None,
            )
            self.resource.publish(flow)

    def test_delete_not_implemented(self):
        """Test that delete raises NotImplementedError for V2."""
        with pytest.raises(NotImplementedError):
            self.resource.delete(flow_id=1)


class TestFlowsCombined(TestAPIBase):
    """Test combined functionality and fallback between V1 and V2."""

    def setUp(self):
        super().setUp()
        # Set up V1 client
        self.v1_http_client = self._get_http_client(
            server=self.server,
            base_url="api/v1/xml",
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        # Set up V2 client
        self.v2_http_client = self._get_http_client(
            server="http://127.0.0.1:8001/",
            base_url="",
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        
        self.resource_v1 = FlowsV1(self.v1_http_client)
        self.resource_v2 = FlowsV2(self.v2_http_client)
        self.resource_fallback = FallbackProxy(self.resource_v2, self.resource_v1)

    # @pytest.mark.skip(reason="V2 API not yet deployed on test server")
    @pytest.mark.uses_test_server()
    def test_get_matches(self):
        """Test that V1 and V2 get methods return matching flow data."""
        flow_id = 1
        
        flow_v1 = self.resource_v1.get(flow_id=flow_id)
        flow_v2 = self.resource_v2.get(flow_id=flow_id)
        
        # Check that the core attributes match
        assert flow_v1.flow_id == flow_v2.flow_id
        assert flow_v1.name == flow_v2.name
        assert flow_v1.version == flow_v2.version
        assert flow_v1.external_version == flow_v2.external_version
        assert flow_v1.description == flow_v2.description

    # @pytest.mark.skip(reason="V2 API not yet deployed on test server")
    @pytest.mark.uses_test_server()
    def test_exists_matches(self):
        """Test that V1 and V2 exists methods return consistent results."""
        # Get a known flow
        flow_v1 = self.resource_v1.get(flow_id=1)
        
        result_v1 = self.resource_v1.exists(
            name=flow_v1.name,
            external_version=flow_v1.external_version
        )
        result_v2 = self.resource_v2.exists(
            name=flow_v1.name,
            external_version=flow_v1.external_version
        )
        
        assert result_v1 is not False
        assert result_v2 is not False
        
        if isinstance(result_v1, int) and isinstance(result_v2, int):
            assert result_v1 == result_v2

    # @pytest.mark.skip(reason="V2 API not yet deployed on test server - fallback would work but tries V2 first")
    @pytest.mark.uses_test_server()
    def test_fallback_get(self):
        """Test that fallback proxy can get flows."""
        flow = self.resource_fallback.get(flow_id=1)
        
        assert isinstance(flow, OpenMLFlow)
        assert flow.flow_id == 1

    # @pytest.mark.skip(reason="V2 API not yet deployed on test server - fallback would work but tries V2 first")
    @pytest.mark.uses_test_server()
    def test_fallback_exists(self):
        """Test that fallback proxy can check flow existence."""
        flow = self.resource_fallback.get(flow_id=1)
        
        result = self.resource_fallback.exists(
            name=flow.name,
            external_version=flow.external_version
        )
        
        assert result is not False

    @pytest.mark.uses_test_server()
    def test_fallback_list_falls_back_to_v1(self):
        """Test that fallback proxy falls back to V1 for list method."""

        flows_df = self.resource_fallback.list(limit=10)
        
        assert len(flows_df) > 0
        assert len(flows_df) <= 10
        assert "id" in flows_df.columns

    def test_fallback_raises_when_all_not_implemented(self):
        """Test that fallback proxy raises NotImplementedError when all APIs raise it."""
        # Both V2 and a hypothetical V1 that doesn't support something should raise
        # For now, we can't easily test this without mocking, but document the behavior
        pass
