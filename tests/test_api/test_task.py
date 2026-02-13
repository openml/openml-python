from __future__ import annotations

import pytest
import pandas as pd
from openml._api.resources.task import TaskV1API, TaskV2API
from openml._api.resources.base.fallback import FallbackProxy
from openml.exceptions import OpenMLNotSupportedError
from openml.testing import TestAPIBase
from openml.enums import APIVersion
from openml.tasks.task import TaskType


class TestTaskAPIBase(TestAPIBase):
    """Common utilities for Task API tests."""
    def _get_first_tid(self, api_resource, task_type: TaskType) -> int:
        tasks = api_resource.list(limit=1, offset=0, task_type=task_type)
        if tasks.empty:
            pytest.skip(f"No tasks of type {task_type} found.")
        return int(tasks.iloc[0]["tid"])

class TestTaskV1API(TestTaskAPIBase):
    def setUp(self):
        super().setUp()
        self.client = self.http_clients[APIVersion.V1]
        self.task = TaskV1API(self.client)

    @pytest.mark.uses_test_server()
    def test_list_tasks(self):
        """Verify V1 list endpoint returns a populated DataFrame."""
        tasks_df = self.task.list(limit=5, offset=0)
        assert isinstance(tasks_df, pd.DataFrame)
        assert not tasks_df.empty
        assert "tid" in tasks_df.columns

class TestTaskV2API(TestTaskAPIBase):
    def setUp(self):
        super().setUp()
        self.client = self.http_clients[APIVersion.V2]
        self.task = TaskV2API(self.client)

    @pytest.mark.uses_test_server()
    def test_list_tasks(self):
        """Verify V2 list endpoint returns a populated DataFrame."""
        with pytest.raises(OpenMLNotSupportedError):
            self.task.list(limit=5, offset=0)

class TestTasksCombined(TestTaskAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self.http_clients[APIVersion.V1]
        self.v2_client = self.http_clients[APIVersion.V2]
        self.task_v1 = TaskV1API(self.v1_client)
        self.task_v2 = TaskV2API(self.v2_client)
        self.task_fallback = FallbackProxy(self.task_v1, self.task_v2)

    def _get_first_tid(self, task_type: TaskType) -> int:
        """Helper to find an existing task ID for a given type using the V1 resource."""
        tasks = self.task_v1.list(limit=1, offset=0, task_type=task_type)
        if tasks.empty:
            pytest.skip(f"No tasks of type {task_type} found on test server.")
        return int(tasks.iloc[0]["tid"])

    @pytest.mark.uses_test_server()
    def test_get_matches(self):
        """Verify that we can get a task from V2 API and it matches V1."""
        tid = self._get_first_tid(TaskType.SUPERVISED_CLASSIFICATION)
        
        output_v1 = self.task_v1.get(tid)
        output_v2 = self.task_v2.get(tid)

        assert int(output_v1.task_id) == tid
        assert int(output_v2.task_id) == tid
        assert output_v1.task_id == output_v2.task_id
        assert output_v1.task_type == output_v2.task_type

    @pytest.mark.uses_test_server()
    def test_get_fallback(self):
        """Verify the fallback proxy works for retrieving tasks."""
        tid = self._get_first_tid(TaskType.SUPERVISED_CLASSIFICATION)
        output_fallback = self.task_fallback.get(tid)
        assert int(output_fallback.task_id) == tid