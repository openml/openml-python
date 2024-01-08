# License: BSD 3-Clause
from __future__ import annotations

import ast

import numpy as np

import openml
from openml.exceptions import OpenMLServerException
from openml.tasks import TaskType
from openml.testing import TestBase, check_task_existence

from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLRegressionTaskTest(OpenMLSupervisedTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super().setUp()

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
                TestBase.logger.info(f"collected from test_run_functions: {task_id}")
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
        X, Y = super().test_get_X_and_Y()
        assert X.shape == (194, 32)
        assert isinstance(X, np.ndarray)
        assert Y.shape == (194,)
        assert isinstance(Y, np.ndarray)
        assert Y.dtype == float

    def test_download_task(self):
        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.SUPERVISED_REGRESSION
        assert task.dataset_id == 105
