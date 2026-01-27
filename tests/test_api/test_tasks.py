# License: BSD 3-Clause
from __future__ import annotations

from openml._api.config import settings
import pytest
import pandas as pd
from openml._api.clients.http import HTTPClient
from openml.testing import TestBase
from openml._api.resources.tasks import TasksV1, TasksV2
from openml.tasks.task import (
    OpenMLClassificationTask, 
    OpenMLRegressionTask, 
    OpenMLLearningCurveTask,
    TaskType
)

class TestTasksEndpoints(TestBase):
    def setUp(self):
        super().setUp()
        v1_http_client = HTTPClient(
            server=settings.api.v1.server,
            base_url=settings.api.v1.base_url,
            api_key=settings.api.v1.api_key,
            timeout=settings.api.v1.timeout,
            retries=settings.connection.retries,
            delay_method=settings.connection.delay_method,
            delay_time=settings.connection.delay_time,
        )
        v2_http_client = HTTPClient(
            server=settings.api.v2.server,
            base_url=settings.api.v2.base_url,
            api_key=settings.api.v2.api_key,
            timeout=settings.api.v2.timeout,
            retries=settings.connection.retries,
            delay_method=settings.connection.delay_method,
            delay_time=settings.connection.delay_time,
        )
        self.v1_api = TasksV1(v1_http_client)
        self.v2_api = TasksV2(v2_http_client)

    def _get_first_tid(self, task_type: TaskType) -> int:
        """Helper to find an existing task ID for a given type on the server."""
        tasks = self.v1_api.list(limit=1, offset=0, task_type=task_type)
        if tasks.empty:
            pytest.skip(f"No tasks of type {task_type} found on test server.")
        return int(tasks.iloc[0]["tid"])

    @pytest.mark.uses_test_server()
    def test_v1_get_classification_task(self):
        tid = self._get_first_tid(TaskType.SUPERVISED_CLASSIFICATION)
        task = self.v1_api.get(tid)
        assert isinstance(task, OpenMLClassificationTask)
        assert int(task.task_id) == tid

    @pytest.mark.uses_test_server()
    def test_v1_get_regression_task(self):
        tid = self._get_first_tid(TaskType.SUPERVISED_REGRESSION)
        task = self.v1_api.get(tid)
        assert isinstance(task, OpenMLRegressionTask)
        assert int(task.task_id) == tid

    @pytest.mark.uses_test_server()
    def test_v1_get_learning_curve_task(self):
        tid = self._get_first_tid(TaskType.LEARNING_CURVE)
        task = self.v1_api.get(tid)
        assert isinstance(task, OpenMLLearningCurveTask)
        assert int(task.task_id) == tid

    @pytest.mark.uses_test_server()
    def test_v1_list_tasks(self):
        """Verify V1 list endpoint returns a populated DataFrame."""
        tasks_df = self.v1_api.list(limit=5, offset=0)
        assert isinstance(tasks_df, pd.DataFrame)
        assert not tasks_df.empty
        assert "tid" in tasks_df.columns

    @pytest.mark.uses_test_server()
    def test_v2_get_task(self):
        """Verify TasksV2 (JSON) skips gracefully if V2 is not supported."""
        tid = self._get_first_tid(TaskType.SUPERVISED_CLASSIFICATION)
        task_v2 = self.v2_api.get(tid)
        assert int(task_v2.task_id) == tid

    @pytest.mark.uses_test_server()
    def test_v1_estimation_procedure_list(self):
        procs = self.v1_api._get_estimation_procedure_list()
        assert isinstance(procs, list)
        assert len(procs) > 0
        assert "id" in procs[0]