# License: BSD 3-Clause
from __future__ import annotations

import pandas as pd
import pytest

from openml.tasks import TaskType, get_task

from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLClassificationTaskTest(OpenMLSupervisedTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super().setUp()
        self.task_id = 119  # diabetes
        self.task_type = TaskType.SUPERVISED_CLASSIFICATION
        self.estimation_procedure = 5

    @pytest.mark.test_server()
    def test_download_task(self):
        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION
        assert task.dataset_id == 20
        assert task.estimation_procedure_id == self.estimation_procedure

    @pytest.mark.test_server()
    def test_class_labels(self):
        task = get_task(self.task_id)
        assert task.class_labels == ["tested_negative", "tested_positive"]


@pytest.mark.test_server()
def test_get_X_and_Y():
    task = get_task(119)
    X, Y = task.get_X_and_y()
    assert X.shape == (768, 8)
    assert isinstance(X, pd.DataFrame)
    assert Y.shape == (768,)
    assert isinstance(Y, pd.Series)
    assert pd.api.types.is_categorical_dtype(Y)
