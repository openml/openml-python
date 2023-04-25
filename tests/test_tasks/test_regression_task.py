# License: BSD 3-Clause

import ast
import numpy as np

import openml
from openml.tasks import TaskType
from openml.testing import TestBase
from openml.testing import check_task_existence
from openml.exceptions import OpenMLServerException
from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLRegressionTaskTest(OpenMLSupervisedTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super(OpenMLRegressionTaskTest, self).setUp()

        task_meta_data = {
            "task_type": TaskType.SUPERVISED_REGRESSION,
            "dataset_id": 105,  # wisconsin
            "estimation_procedure_id": 7,
            "target_name": "time",
        }
        _task_id = check_task_existence(**task_meta_data)
        if _task_id is not None:
            task_id = _task_id
        else:
            new_task = openml.tasks.create_task(**task_meta_data)
            # publishes the new task
            try:
                new_task = new_task.publish()
                task_id = new_task.task_id
                # mark to remove the uploaded task
                TestBase._mark_entity_for_removal("task", task_id)
                TestBase.logger.info("collected from test_run_functions: {}".format(task_id))
            except OpenMLServerException as e:
                if e.code == 614:  # Task already exists
                    # the exception message contains the task_id that was matched in the format
                    # 'Task already exists. - matched id(s): [xxxx]'
                    task_id = ast.literal_eval(e.message.split("matched id(s):")[-1].strip())[0]
                else:
                    raise Exception(repr(e))
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
