# License: BSD 3-Clause
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import requests

from openml.tasks import TaskType, get_task
from openml.testing import create_request_response

from .test_supervised_task import OpenMLSupervisedTaskTest

_MOCK_DIR = Path(__file__).parent.parent / "files" / "mock_responses"

_TASK_RESPONSES = [
    create_request_response(status_code=200, content_filepath=_MOCK_DIR / "tasks" / "task_description_119.xml"),
    create_request_response(status_code=200, content_filepath=_MOCK_DIR / "datasets" / "data_description_20.xml"),
    create_request_response(status_code=200, content_filepath=_MOCK_DIR / "datasets" / "data_features_20.xml"),
    create_request_response(status_code=200, content_filepath=_MOCK_DIR / "datasets" / "diabetes.arff"),
]


class OpenMLClassificationTaskTest(OpenMLSupervisedTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super().setUp()
        self.task_id = 119  # diabetes
        self.task_type = TaskType.SUPERVISED_CLASSIFICATION
        self.estimation_procedure = 5

    @mock.patch.object(requests.Session, "get")
    def test_download_task(self, mock_get):
        mock_get.side_effect = _TASK_RESPONSES[:3]
        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION
        assert task.dataset_id == 20
        assert task.estimation_procedure_id == self.estimation_procedure

    @mock.patch.object(requests.Session, "get")
    def test_class_labels(self, mock_get):
        mock_get.side_effect = _TASK_RESPONSES[:3]
        task = get_task(self.task_id)
        assert task.class_labels == ["tested_negative", "tested_positive"]

    @mock.patch.object(requests.Session, "get")
    def test_get_X_and_Y(self, mock_get):
        mock_get.side_effect = _TASK_RESPONSES
        task = get_task(self.task_id)
        X, Y = task.get_X_and_y()
        assert X.shape == (768, 8)
        assert isinstance(X, pd.DataFrame)
        assert Y.shape == (768,)
        assert isinstance(Y, pd.Series)
        assert pd.api.types.is_categorical_dtype(Y)
