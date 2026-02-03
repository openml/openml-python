"""Tests for Run V1 → V2 API Migration."""
from __future__ import annotations

import pytest

import openml
from openml._api.resources import FallbackProxy, RunV1API, RunV2API
from openml.exceptions import OpenMLNotSupportedError
from openml.runs.run import OpenMLRun
from openml.testing import TestAPIBase


class TestRunsV1(TestAPIBase):
	"""Test RunsV1 resource implementation."""

	def setUp(self):
		super().setUp()
		self.resource = RunV1API(self.http_client)

	@pytest.mark.uses_test_server()
	def test_get(self):
		"""Test getting a run from the V1 API."""
		run = self.resource.get(run_id=1)

		assert isinstance(run, OpenMLRun)
		assert run.run_id == 1
		assert isinstance(run.task_id, int)

	@pytest.mark.uses_test_server()
	def test_list(self):
		"""Test listing runs from the V1 API."""
		runs_df = self.resource.list(limit=5, offset=0)

		assert len(runs_df) > 0
		assert len(runs_df) <= 5
		assert "run_id" in runs_df.columns
		assert "task_id" in runs_df.columns
		assert "setup_id" in runs_df.columns
		assert "flow_id" in runs_df.columns

	@pytest.mark.uses_test_server()
	def test_publish(self):
		"""Test publishing a small run using V1 API."""
		from sklearn.neighbors import KNeighborsClassifier

		task = openml.tasks.get_task(19)
		clf = KNeighborsClassifier(n_neighbors=3)
		run = openml.runs.run_model_on_task(clf, task)

		file_elements = run._get_file_elements()
		if "description" not in file_elements:
			file_elements["description"] = run._to_xml()

		run_id = self.resource.publish(path="run", files=file_elements)
		assert isinstance(run_id, int)
		assert run_id > 0

	@pytest.mark.uses_test_server()
	def test_delete_run(self):
		"""Test deleting a run using V1 API."""
		# First, create and publish a run to delete
		from sklearn.neighbors import KNeighborsClassifier

		task = openml.tasks.get_task(19)
		clf = KNeighborsClassifier(n_neighbors=3)
		run = openml.runs.run_model_on_task(clf, task)

		file_elements = run._get_file_elements()
		if "description" not in file_elements:
			file_elements["description"] = run._to_xml()

		run_id = self.resource.publish(path="run", files=file_elements)
		assert isinstance(run_id, int)
		assert run_id > 0

		# Now delete the run
		self.resource.delete(run_id)

		# Verify deletion by attempting to fetch the run
		with pytest.raises(Exception):
			self.resource.get(run_id=run_id)


class TestRunsV2(TestAPIBase):
	"""Test RunsV2 resource implementation."""

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
		self.resource = RunV2API(self.v2_http_client)

	@pytest.mark.uses_test_server()
	def test_get_not_supported(self):
		"""Test that V2 get is not implemented."""
		with pytest.raises(OpenMLNotSupportedError):
			_ = self.resource.get(run_id=1)

	@pytest.mark.uses_test_server()
	def test_list_not_supported(self):
		"""Test that V2 list is not implemented."""
		with pytest.raises(OpenMLNotSupportedError):
			_ = self.resource.list(limit=5, offset=0)


class TestRunsCombined(TestAPIBase):
	"""Test fallback behavior between V2 and V1 for Runs."""

	def setUp(self):
		super().setUp()
		self.v1_client = self._get_http_client(
			server=self.server,
			base_url=self.base_url,
			api_key=self.api_key,
			timeout=self.timeout,
			retries=self.retries,
			retry_policy=self.retry_policy,
			cache=self.cache,
		)
		self.v2_client = self._get_http_client(
			server="http://127.0.0.1:8001/",
			base_url="",
			api_key=self.api_key,
			timeout=self.timeout,
			retries=self.retries,
			retry_policy=self.retry_policy,
			cache=self.cache,
		)

		self.resource_v1 = RunV1API(self.v1_client)
		self.resource_v2 = RunV2API(self.v2_client)
		self.resource_fallback = FallbackProxy(self.resource_v2, self.resource_v1)

	@pytest.mark.uses_test_server()
	def test_get_fallback(self):
		"""Test fallback for get() when V2 is not implemented."""
		run = self.resource_fallback.get(run_id=1)
		assert isinstance(run, OpenMLRun)
		assert run.run_id == 1

	@pytest.mark.uses_test_server()
	def test_list_fallback(self):
		"""Test fallback for list() when V2 is not implemented."""
		runs_df = self.resource_fallback.list(limit=5, offset=0)
		assert len(runs_df) > 0
		assert "run_id" in runs_df.columns
