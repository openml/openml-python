import unittest
from typing import Tuple

import numpy as np

import openml
from tests.test_tasks import OpenMLTaskTest


class OpenMLSupervisedTaskTest(OpenMLTaskTest):
    """
    A helper class. The methods of the test case
    are only executed in subclasses of the test case.
    """
    def setUp(self):

        super(OpenMLSupervisedTaskTest, self).setUp()
        # task_id acts as a placeholder variable
        # and it is set from the extending classes.
        self.task_id = 1

    @classmethod
    def setUpClass(cls):
        if cls is OpenMLSupervisedTaskTest:
            raise unittest.SkipTest(
                "Skip OpenMLSupervisedTaskTest tests,"
                " it's a base class"
            )
        super(OpenMLSupervisedTaskTest, cls).setUpClass()

    def test_get_X_and_Y(self) -> Tuple[np.ndarray, np.ndarray]:

        task = openml.tasks.get_task(self.task_id)
        X, Y = task.get_X_and_y()
        return X, Y
