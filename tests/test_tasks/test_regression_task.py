# License: BSD 3-Clause
from __future__ import annotations

from unittest import mock

import pandas as pd

from openml.tasks import OpenMLRegressionTask, TaskType

from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLRegressionTaskTest(OpenMLSupervisedTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super().setUp()
        self.estimation_procedure = 9
        self.task_id = 1000
        self.task_type = TaskType.SUPERVISED_REGRESSION

    @mock.patch("tests.test_tasks.test_supervised_task.get_task")
    def test_get_X_and_Y(self, mock_get_task):
        mock_task = mock.MagicMock()
        X = pd.DataFrame(0.0, index=range(194), columns=range(32))
        Y = pd.Series(0.0, index=range(194), name="time", dtype=float)
        mock_task.get_X_and_y.return_value = (X, Y)
        mock_get_task.return_value = mock_task
        X, Y = super().test_get_X_and_Y()
        assert X.shape == (194, 32)
        assert isinstance(X, pd.DataFrame)
        assert Y.shape == (194,)
        assert isinstance(Y, pd.Series)
        assert pd.api.types.is_numeric_dtype(Y)

    @mock.patch("tests.test_tasks.test_task.get_task")
    def test_download_task(self, mock_get_task):
        task = OpenMLRegressionTask(
            task_type_id=TaskType.SUPERVISED_REGRESSION,
            task_type="Supervised Regression",
            data_set_id=105,
            target_name="time",
            estimation_procedure_id=self.estimation_procedure,
            task_id=self.task_id,
        )
        mock_get_task.return_value = task
        result = super().test_download_task()
        assert result.task_id == self.task_id
        assert result.task_type_id == TaskType.SUPERVISED_REGRESSION
        assert result.dataset_id == 105
        assert result.estimation_procedure_id == self.estimation_procedure
