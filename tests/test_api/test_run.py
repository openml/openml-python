"""Tests for Run V1 → V2 API Migration."""
from __future__ import annotations

import pytest

import openml
from openml._api.resources import FallbackProxy, RunV1API, RunV2API
from openml.enums import APIVersion
from openml.exceptions import OpenMLNotSupportedError
from openml.runs.run import OpenMLRun
from openml.testing import TestAPIBase


@pytest.mark.uses_test_server()
class TestRunsV1(TestAPIBase):
	"""Test RunsV1 resource implementation."""

	def setUp(self):
		super().setUp()
		http_client = self.http_clients[APIVersion.V1]
		self.resource = RunV1API(http_client)

	def test_get(self):
		"""Test getting a run from the V1 API."""
		run = self.resource.get(run_id=1)

		self.assertIsInstance(run, OpenMLRun)
		self.assertEqual(run.run_id, 1)
		self.assertIsInstance(run.task_id, int)

	def test_list(self):
		"""Test listing runs from the V1 API."""
		limit = 5
		runs_df = self.resource.list(limit=limit, offset=0)

		self.assertEqual(len(runs_df), limit)
		self.assertIn("run_id", runs_df.columns)
		self.assertIn("task_id", runs_df.columns)
		self.assertIn("setup_id", runs_df.columns)
		self.assertIn("flow_id", runs_df.columns)

	def test_delete_and_publish_run(self):
		"""Test publishing then deleting a run using V1 API."""
		# First, create and publish a run to delete
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


@pytest.mark.uses_test_server()
class TestRunsV2(TestRunsV1):
	"""Test RunsV2 resource implementation."""

	def setUp(self):
		super().setUp()
		http_client = self.http_clients[APIVersion.V2]
		self.resource = RunV2API(http_client)

	def test_get(self):
		with pytest.raises(OpenMLNotSupportedError):
			super().test_get()

	def test_list(self):
		with pytest.raises(OpenMLNotSupportedError):
			super().test_list()

	def test_delete_and_publish_run(self):
		with pytest.raises(OpenMLNotSupportedError):
			super().test_delete_and_publish_run()


@pytest.mark.uses_test_server()
class TestRunsFallback(TestRunsV1):
	"""Test combined functionality and fallback between V1 and V2."""

	def setUp(self):
		super().setUp()
		http_client_v1 = self.http_clients[APIVersion.V1]
		resource_v1 = RunV1API(http_client_v1)

		http_client_v2 = self.http_clients[APIVersion.V2]
		resource_v2 = RunV2API(http_client_v2)

		self.resource = FallbackProxy(resource_v2, resource_v1)
