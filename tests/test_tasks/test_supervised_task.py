import unittest
from typing import Tuple

import numpy as np

import openml
from tests.test_tasks import OpenMLTaskTest

# Helper class
# The test methods in this class
# are not supposed to be executed.
class OpenMLSupervisedTaskTest(OpenMLTaskTest):
    # task id will be set from the
    # extending classes

    def setUp(self):

        super(OpenMLSupervisedTaskTest, self).setUp()
        self.task_id = 1

    @classmethod
    def setUpClass(cls):
        if cls is OpenMLSupervisedTaskTest:
            raise unittest.SkipTest(
                "Skip OpenMLSupervisedTaskTest tests,"
                " it's a base class"
            )
        super(OpenMLSupervisedTaskTest, cls).setUpClass()

    def test_get_X_and_Y(self) -> \
            Tuple[np.ndarray, np.ndarray]:

        task = openml.tasks.get_task(self.task_id)
        X, Y = task.get_X_and_y()
        return X, Y
