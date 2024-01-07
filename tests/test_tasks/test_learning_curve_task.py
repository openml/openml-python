# License: BSD 3-Clause
from __future__ import annotations

import numpy as np

from openml.tasks import TaskType, get_task

from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLLearningCurveTaskTest(OpenMLSupervisedTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super().setUp()
        self.task_id = 801  # diabetes
        self.task_type = TaskType.LEARNING_CURVE
        self.estimation_procedure = 13

    def test_get_X_and_Y(self):
        X, Y = super().test_get_X_and_Y()
        assert X.shape == (768, 8)
        assert isinstance(X, np.ndarray)
        assert Y.shape == (768,)
        assert isinstance(Y, np.ndarray)
        assert Y.dtype == int

    def test_download_task(self):
        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.LEARNING_CURVE
        assert task.dataset_id == 20

    def test_class_labels(self):
        task = get_task(self.task_id)
        assert task.class_labels == ["tested_negative", "tested_positive"]
