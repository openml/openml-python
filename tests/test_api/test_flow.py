# License: BSD 3-Clause
"""Tests for Flow V1 → V2 API Migration."""
from __future__ import annotations

import uuid

import pytest

from openml._api.resources import FallbackProxy, FlowV1API, FlowV2API
from openml.enums import APIVersion
from openml.exceptions import OpenMLNotSupportedError
from openml.flows.flow import OpenMLFlow
from openml.testing import TestAPIBase

@pytest.mark.uses_test_server()
class TestFlowsV1(TestAPIBase):
    """Test FlowsV1 resource implementation."""

    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V1]
        self.resource = FlowV1API(http_client)

    def test_get(self):
        """Test getting a flow from the V1 API."""
        flow = self.resource.get(flow_id=1)
        
        self.assertIsInstance(flow, OpenMLFlow)
        self.assertEqual(flow.flow_id, 1)
        self.assertIsInstance(flow.name, str)
        self.assertGreater(len(flow.name), 0)

    def test_exists(self):
        """Test checking if a flow exists using V1 API."""
        flow = self.resource.get(flow_id=1)
        
        result = self.resource.exists(
            name=flow.name,
            external_version=flow.external_version
        )
        
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        self.assertEqual(result, flow.flow_id)

    def test_exists_nonexistent(self):
        """Test checking if a non-existent flow exists using V1 API."""
        result = self.resource.exists(
            name="NonExistentFlowName123456789",
            external_version="0.0.0.nonexistent"
        )
        
        self.assertFalse(result)

    def test_list(self):
        """Test listing flows from the V1 API."""
        limit = 10
        flows_df = self.resource.list(limit=limit)
        
        self.assertEqual(len(flows_df), limit)
        self.assertIn("id", flows_df.columns)
        self.assertIn("name", flows_df.columns)
        self.assertIn("version", flows_df.columns)
        self.assertIn("external_version", flows_df.columns)
        self.assertIn("full_name", flows_df.columns)
        self.assertIn("uploader", flows_df.columns)

    def test_list_with_offset(self):
        """Test listing flows with offset from the V1 API."""
        limit = 5
        flows_df = self.resource.list(limit=limit, offset=10)
        
        self.assertEqual(len(flows_df), limit)

    def test_list_with_tag_limit_offset(self):
        """Test listing flows with filters from the V1 API."""
        limit = 5
        flows_df = self.resource.list(tag="weka", limit=limit, offset=0, uploader=16)
        
        self.assertTrue(hasattr(flows_df, "columns"))
        self.assertLessEqual(len(flows_df), limit)
        if len(flows_df) > 0:
            self.assertIn("id", flows_df.columns)

    def test_delete_and_publish(self):
        """Test deleting a flow using V1 API."""
        from openml_sklearn.extension import SklearnExtension
        from sklearn.tree import ExtraTreeRegressor

        clf = ExtraTreeRegressor()
        extension = SklearnExtension()
        dt_flow = extension.model_to_flow(clf)

        # Check if flow exists, if not publish it
        flow_id = self.resource.exists(
            name=dt_flow.name,
            external_version=dt_flow.external_version,
        )
        
        if not flow_id:
            # Publish the flow first
            file_elements = dt_flow._get_file_elements()
            if "description" not in file_elements:
                file_elements["description"] = dt_flow._to_xml()
            
            flow_id = self.resource.publish(files=file_elements)
        
        # Now delete it
        result = self.resource.delete(flow_id)
        self.assertTrue(result)

        # Verify it no longer exists
        exists = self.resource.exists(
            name=dt_flow.name,
            external_version=dt_flow.external_version,
        )
        self.assertFalse(exists)

@pytest.mark.uses_test_server()
class TestFlowsV2(TestFlowsV1):
    """Test FlowsV2 resource implementation."""

    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V2]
        self.resource = FlowV2API(http_client)

    def test_list(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_list()

    def test_list_with_offset(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_list_with_offset()

    def test_list_with_tag_limit_offset(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_list_with_tag_limit_offset()

    def test_delete_and_publish(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_delete_and_publish()

@pytest.mark.uses_test_server()
class TestFlowsFallback(TestFlowsV1):
    """Test combined functionality and fallback between V1 and V2."""

    def setUp(self):
        super().setUp()
        http_client_v1 = self.http_clients[APIVersion.V1]
        resource_v1 = FlowV1API(http_client_v1)

        http_client_v2 = self.http_clients[APIVersion.V2]
        resource_v2 = FlowV2API(http_client_v2)
        
        self.resource = FallbackProxy(resource_v2, resource_v1)


