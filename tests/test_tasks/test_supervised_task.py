from typing import Tuple
import unittest

import numpy as np

from openml.tasks import get_task
from .test_task import OpenMLTaskTest


class OpenMLSupervisedTaskTest(OpenMLTaskTest):
    """
    A helper class. The methods of the test case
    are only executed in subclasses of the test case.
    """
    @classmethod
    def setUpClass(cls):
        if cls is OpenMLSupervisedTaskTest:
            raise unittest.SkipTest(
                "Skip OpenMLSupervisedTaskTest tests,"
                " it's a base class"
            )
        super(OpenMLSupervisedTaskTest, cls).setUpClass()

    def setUp(self, n_levels: int = 1):
        super(OpenMLSupervisedTaskTest, self).setUp()
        self.task_id = None
        self.task_type_id = None
        self.estimation_procedure = None

    def test_get_X_and_Y(self) -> Tuple[np.ndarray, np.ndarray]:

        task = get_task(self.task_id)
        X, Y = task.get_X_and_y()
        return X, Y
