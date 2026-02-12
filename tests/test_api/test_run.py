"""Tests for Run V1 → V2 API Migration."""
from __future__ import annotations

import pytest

import openml
from openml._api.resources import RunV1API, RunV2API
from openml.enums import APIVersion
from openml.exceptions import OpenMLNotSupportedError
from openml.runs.run import OpenMLRun
from openml.testing import TestAPIBase


@pytest.mark.uses_test_server()
class TestRunAPIBase(TestAPIBase):
	resource: RunV1API | RunV2API

	def _assert_run_shape(self, run: OpenMLRun) -> None:
		self.assertIsInstance(run, OpenMLRun)
		self.assertEqual(run.run_id, 1)
		self.assertIsInstance(run.task_id, int)

	def _get(self) -> OpenMLRun:
		run = self.resource.get(run_id=1)
		self._assert_run_shape(run)
		return run

	def _list(self) -> None:
		limit = 5
		runs_df = self.resource.list(limit=limit, offset=0)

		self.assertEqual(len(runs_df), limit)
		self.assertIn("run_id", runs_df.columns)
		self.assertIn("task_id", runs_df.columns)
		self.assertIn("setup_id", runs_df.columns)
		self.assertIn("flow_id", runs_df.columns)

	def _publish_and_delete(self) -> None:
		from sklearn.neighbors import KNeighborsClassifier

		task = openml.tasks.get_task(19)
		clf = KNeighborsClassifier(n_neighbors=3)
		run = openml.runs.run_model_on_task(clf, task)

		file_elements = run._get_file_elements()
		if "description" not in file_elements:
			file_elements["description"] = run._to_xml()

		run_id = self.resource.publish(path="run", files=file_elements)
		self.assertIsInstance(run_id, int)
		self.assertGreater(run_id, 0)

		self.resource.delete(run_id)

		with pytest.raises(Exception):
			self.resource.get(run_id=run_id)


class TestRunV1API(TestRunAPIBase):
	def setUp(self):
		super().setUp()
		http_client = self.http_clients[APIVersion.V1]
		self.resource = RunV1API(http_client)

	def test_get(self):
		self._get()

	def test_list(self):
		self._list()

	def test_publish_and_delete(self):
		self._publish_and_delete()


class TestRunV2API(TestRunAPIBase):
	def setUp(self):
		super().setUp()
		http_client = self.http_clients[APIVersion.V2]
		self.resource = RunV2API(http_client)

	def test_get(self):
		with pytest.raises(
			OpenMLNotSupportedError,
			match="RunV2API: v2 API does not support `get` for resource `run`",
		):
			self._get()

	def test_list(self):
		with pytest.raises(
			OpenMLNotSupportedError,
			match="RunV2API: v2 API does not support `list` for resource `run`",
		):
			self._list()

	def test_publish_and_delete(self):
		with pytest.raises(
			OpenMLNotSupportedError,
			match="RunV2API: v2 API does not support `publish` for resource `run`",
		):
			self._publish_and_delete()


class TestRunCombinedAPI(TestAPIBase):
	def setUp(self):
		super().setUp()
		self.resource_v1 = RunV1API(self.http_clients[APIVersion.V1])
		self.resource_v2 = RunV2API(self.http_clients[APIVersion.V2])

	def test_get_contracts(self):
		run_v1 = self.resource_v1.get(run_id=1)
		self.assertIsInstance(run_v1, OpenMLRun)
		self.assertEqual(run_v1.run_id, 1)

		with pytest.raises(
			OpenMLNotSupportedError,
			match="RunV2API: v2 API does not support `get` for resource `run`",
		):
			self.resource_v2.get(run_id=1)

	def test_list_contracts(self):
		limit = 5
		runs_v1 = self.resource_v1.list(limit=limit, offset=0)
		self.assertEqual(len(runs_v1), limit)
		self.assertIn("run_id", runs_v1.columns)
		self.assertIn("task_id", runs_v1.columns)
		self.assertIn("setup_id", runs_v1.columns)
		self.assertIn("flow_id", runs_v1.columns)

		with pytest.raises(
			OpenMLNotSupportedError,
			match="RunV2API: v2 API does not support `list` for resource `run`",
		):
			self.resource_v2.list(limit=limit, offset=0)

	def test_publish_contracts(self):
		with pytest.raises(
			OpenMLNotSupportedError,
			match="RunV2API: v2 API does not support `publish` for resource `run`",
		):
			self.resource_v2.publish(path="run", files={"description": "<run/>"})
