# License: BSD 3-Clause
from __future__ import annotations

import unittest

import pandas as pd

from openml.tasks import get_task
import pytest

from .test_task import OpenMLTaskTest


class OpenMLSupervisedTaskTest(OpenMLTaskTest):
    """
    A helper class. The methods of the test case
    are only executed in subclasses of the test case.
    """

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if cls is OpenMLSupervisedTaskTest:
            raise unittest.SkipTest("Skip OpenMLSupervisedTaskTest tests," " it's a base class")
        super().setUpClass()

    def setUp(self, n_levels: int = 1):
        super().setUp()

    @pytest.mark.test_server()
    def test_get_X_and_Y(self) -> tuple[pd.DataFrame, pd.Series]:
        task = get_task(self.task_id)
        X, Y = task.get_X_and_y()
        return X, Y
