# License: BSD 3-Clause
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

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
        mock_task = MagicMock()
        mock_task.get_X_and_y.return_value = (
            pd.DataFrame({"a": [1, 2]}),
            pd.Series([0, 1]),
        )
        with patch("openml.tasks.get_task", return_value=mock_task):
            X, Y = mock_task.get_X_and_y()
        return X, Y
