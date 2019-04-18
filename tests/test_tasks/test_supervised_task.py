from typing import Tuple
import unittest

import numpy as np

from .test_task import OpenMLTaskTest


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
        self.task_type_id = 1

    @classmethod
    def setUpClass(cls):
        super(OpenMLSupervisedTaskTest, cls).setUpClass()
        if cls is OpenMLSupervisedTaskTest:
            raise unittest.SkipTest(
                "Skip OpenMLSupervisedTaskTest tests,"
                " it's a base class"
            )

    def test_get_X_and_Y(self) -> Tuple[np.ndarray, np.ndarray]:

        task = super(OpenMLSupervisedTaskTest, self).test_download_task()
        X, Y = task.get_X_and_y()
        return X, Y
