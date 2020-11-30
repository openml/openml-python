# License: BSD 3-Clause

import numpy as np

import openml
from openml.tasks import TaskType
from openml.testing import TestBase
from openml.utils import check_task_existence
from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLRegressionTaskTest(OpenMLSupervisedTaskTest):

    __test__ = True

    def setUp(self, n_levels: int = 1):
        super(OpenMLRegressionTaskTest, self).setUp()

        task_id = 1734
        task_meta_data = {
            "task_type": "Supervised Regression",
            "dataset_id": 105,
            "estimation_procedure_id": 7,
            "target_name": "time",
        }
        if not check_task_existence(task_id, task_meta_data):
            task_meta_data["task_type"] = TaskType.SUPERVISED_REGRESSION
            new_task = openml.tasks.create_task(**task_meta_data)
            # publishes the new task
            new_task = new_task.publish()
            task_id = new_task.task_id
            # mark to remove the uploaded task
            TestBase._mark_entity_for_removal("task", task_id)
            TestBase.logger.info("collected from test_run_functions: {}".format(task_id))

        self.task_id = task_id
        self.task_type = TaskType.SUPERVISED_REGRESSION
        self.estimation_procedure = 7

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLRegressionTaskTest, self).test_get_X_and_Y()
        self.assertEqual((194, 32), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((194,), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, float)

    def test_download_task(self):

        task = super(OpenMLRegressionTaskTest, self).test_download_task()
        self.assertEqual(task.task_id, self.task_id)
        self.assertEqual(task.task_type_id, TaskType.SUPERVISED_REGRESSION)
        self.assertEqual(task.dataset_id, 105)
