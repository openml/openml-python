import unittest

import openml
from tests.test_tasks.test_task import OpenMLTaskTest


@unittest.skip("Supervised class does not need to be tested")
class OpenMLSupervisedTaskTest(OpenMLTaskTest):
    # task id will be set from the
    # extending classes
    def test_get_X_and_Y(self):

        task = openml.tasks.get_task(self.task_id)
        X, Y = task.get_X_and_y()
        return X, Y
