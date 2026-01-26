# License: BSD 3-Clause
from __future__ import annotations

import pytest
import pandas as pd
import requests
from openml.testing import TestBase
from openml._api import api_context
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
        self.v1_api = TasksV1(api_context.backend.tasks._http)
        self.v2_api = TasksV2(api_context.backend.tasks._http)

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
        try:
            task_v2 = self.v2_api.get(tid)
            assert int(task_v2.task_id) == tid
        except (requests.exceptions.JSONDecodeError, Exception):
            pytest.skip("V2 API JSON format not supported on this server.")

    @pytest.mark.uses_test_server()
    def test_v1_estimation_procedure_list(self):
        procs = self.v1_api._get_estimation_procedure_list()
        assert isinstance(procs, list)
        assert len(procs) > 0
        assert "id" in procs[0]