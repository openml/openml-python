# License: BSD 3-Clause
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from openml.tasks import TaskType

from .test_task import OpenMLTaskTest


class OpenMLSupervisedTaskTest(OpenMLTaskTest):
    __test__ = False

    @classmethod
    def setUpClass(cls):
        if cls is OpenMLSupervisedTaskTest:
            raise unittest.SkipTest("Skip OpenMLSupervisedTaskTest tests, it's a base class")
        super().setUpClass()

    def setUp(self, _n_levels: int = 1):
        super().setUp()

    def test_get_X_and_Y(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.task_type == TaskType.SUPERVISED_REGRESSION:
            mock_X = pd.DataFrame({f"f_{i}": [float(i)] * 194 for i in range(32)})
            mock_y = pd.Series([0.0] * 194)
        else:
            mock_X = pd.DataFrame({f"f_{i}": [float(i)] * 768 for i in range(8)})
            mock_y = pd.Series(["tested_negative"] * 768, dtype="category")

        mock_task = MagicMock()
        mock_task.get_X_and_y.return_value = (
            mock_X,
            mock_y,
        )

        with patch("openml.tasks.get_task", return_value=mock_task):
            from openml import tasks as task_module

            task = task_module.get_task(self.task_id)
            X, Y = task.get_X_and_y()
        return X, Y
