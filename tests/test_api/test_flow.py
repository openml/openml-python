# License: BSD 3-Clause
"""Tests for Flow V1 → V2 API Migration."""
from __future__ import annotations

import pytest

from openml._api.resources import FlowV1API, FlowV2API
from openml.enums import APIVersion
from openml.exceptions import OpenMLNotSupportedError
from openml.flows.flow import OpenMLFlow
from openml.testing import TestAPIBase


@pytest.mark.uses_test_server()
class TestFlowAPIBase(TestAPIBase):
    resource: FlowV1API | FlowV2API

    def _assert_flow_shape(self, flow: OpenMLFlow) -> None:
        self.assertIsInstance(flow, OpenMLFlow)
        self.assertEqual(flow.flow_id, 1)
        self.assertIsInstance(flow.name, str)
        self.assertGreater(len(flow.name), 0)

    def _get(self) -> OpenMLFlow:
        flow = self.resource.get(flow_id=1)
        self._assert_flow_shape(flow)
        return flow

    def _exists(self) -> int | bool:
        flow = self.resource.get(flow_id=1)
        result = self.resource.exists(
            name=flow.name,
            external_version=flow.external_version,
        )

        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        self.assertEqual(result, flow.flow_id)
        return result

    def _exists_nonexistent(self) -> int | bool:
        result = self.resource.exists(
            name="NonExistentFlowName123456789",
            external_version="0.0.0.nonexistent",
        )

        self.assertFalse(result)
        return result

    def _list(self) -> None:
        limit = 10
        flows_df = self.resource.list(limit=limit)

        self.assertEqual(len(flows_df), limit)
        self.assertIn("id", flows_df.columns)
        self.assertIn("name", flows_df.columns)
        self.assertIn("version", flows_df.columns)
        self.assertIn("external_version", flows_df.columns)
        self.assertIn("full_name", flows_df.columns)
        self.assertIn("uploader", flows_df.columns)

    def _list_with_offset(self) -> None:
        limit = 5
        flows_df = self.resource.list(limit=limit, offset=10)

        self.assertEqual(len(flows_df), limit)

    def _list_with_tag_limit_offset(self) -> None:
        limit = 5
        flows_df = self.resource.list(tag="weka", limit=limit, offset=0, uploader=16)

        self.assertTrue(hasattr(flows_df, "columns"))
        self.assertLessEqual(len(flows_df), limit)
        if len(flows_df) > 0:
            self.assertIn("id", flows_df.columns)

    def _publish_and_delete(self) -> None:
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
class TestFlowV1API(TestFlowAPIBase):
    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V1]
        self.resource = FlowV1API(http=http_client, minio=self.minio_client)

    def test_get(self):
        self._get()

    def test_exists(self):
        self._exists()

    def test_exists_nonexistent(self):
        self._exists_nonexistent()

    def test_list(self):
        self._list()

    def test_list_with_offset(self):
        self._list_with_offset()

    def test_list_with_tag_limit_offset(self):
        self._list_with_tag_limit_offset()

    def test_publish_and_delete(self):
        self._publish_and_delete()


class TestFlowV2API(TestFlowAPIBase):
    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V2]
        self.resource = FlowV2API(http=http_client, minio=self.minio_client)

    def test_get(self):
        self._get()

    def test_exists(self):
        self._exists()

    def test_exists_nonexistent(self):
        self._exists_nonexistent()

    def test_list(self):
        with pytest.raises(
            OpenMLNotSupportedError,
            match="FlowV2API: v2 API does not support `list` for resource `flow`",
        ):
            self._list()

    def test_list_with_offset(self):
        with pytest.raises(
            OpenMLNotSupportedError,
            match="FlowV2API: v2 API does not support `list` for resource `flow`",
        ):
            self._list_with_offset()

    def test_list_with_tag_limit_offset(self):
        with pytest.raises(
            OpenMLNotSupportedError,
            match="FlowV2API: v2 API does not support `list` for resource `flow`",
        ):
            self._list_with_tag_limit_offset()

    def test_publish_and_delete(self):
        with pytest.raises(
            OpenMLNotSupportedError,
            match="FlowV2API: v2 API does not support `publish` for resource `flow`",
        ):
            self._publish_and_delete()


@pytest.mark.uses_test_server()
class TestFlowCombinedAPI(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.resource_v1 = FlowV1API(
            http=self.http_clients[APIVersion.V1],
            minio=self.minio_client,
        )
        self.resource_v2 = FlowV2API(
            http=self.http_clients[APIVersion.V2],
            minio=self.minio_client,
        )

    def test_get_matches_output(self):
        flow_v1 = self.resource_v1.get(flow_id=1)
        flow_v2 = self.resource_v2.get(flow_id=1)

        self.assertEqual(flow_v1.flow_id, flow_v2.flow_id)
        self.assertEqual(flow_v1.name, flow_v2.name)
        self.assertEqual(flow_v1.version, flow_v2.version)
        self.assertEqual(flow_v1.external_version, flow_v2.external_version)
        self.assertEqual(flow_v1.description, flow_v2.description)

    def test_exists_matches_output(self):
        flow_v1 = self.resource_v1.get(flow_id=1)

        result_v1 = self.resource_v1.exists(
            name=flow_v1.name,
            external_version=flow_v1.external_version,
        )
        result_v2 = self.resource_v2.exists(
            name=flow_v1.name,
            external_version=flow_v1.external_version,
        )

        self.assertIsNot(result_v1, False)
        self.assertIsNot(result_v2, False)
        if isinstance(result_v1, int) and isinstance(result_v2, int):
            self.assertEqual(result_v1, result_v2)

    def test_exists_nonexistent_matches_output(self):
        result_v1 = self.resource_v1.exists(
            name="NonExistentFlowName123456789",
            external_version="0.0.0.nonexistent",
        )
        result_v2 = self.resource_v2.exists(
            name="NonExistentFlowName123456789",
            external_version="0.0.0.nonexistent",
        )

        self.assertFalse(result_v1)
        self.assertFalse(result_v2)

    def test_list_contracts(self):
        with pytest.raises(
            OpenMLNotSupportedError,
            match="FlowV2API: v2 API does not support `list` for resource `flow`",
        ):
            self.resource_v2.list(limit=10)

    def test_publish_contracts(self):
        with pytest.raises(
            OpenMLNotSupportedError,
            match="FlowV2API: v2 API does not support `publish` for resource `flow`",
        ):
            self.resource_v2.publish(path="flow", files={"description": "<flow/>"})


