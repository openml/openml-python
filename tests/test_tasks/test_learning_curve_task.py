# License: BSD 3-Clause
from __future__ import annotations

import requests
from unittest import mock

import pandas as pd
import pytest

import openml
from openml.tasks import TaskType, get_task
from openml.testing import create_request_response

from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLLearningCurveTaskTest(OpenMLSupervisedTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super().setUp()
        self.task_id = 801  # diabetes
        self.task_type = TaskType.LEARNING_CURVE
        self.estimation_procedure = 13

    @pytest.mark.test_server()
    def test_get_X_and_Y(self):
        X, Y = super().test_get_X_and_Y()
        assert X.shape == (768, 8)
        assert isinstance(X, pd.DataFrame)
        assert Y.shape == (768,)
        assert isinstance(Y, pd.Series)
        assert pd.api.types.is_categorical_dtype(Y)

    @pytest.mark.test_server()
    def test_download_task(self):
        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.LEARNING_CURVE
        assert task.dataset_id == 20


@mock.patch.object(requests.Session, "get")
def test_class_labels(mock_get, test_files_directory, test_api_key):
    task_response = create_request_response(
        status_code=200,
        content_filepath=test_files_directory / "mock_responses" / "tasks" / "task_801.xml",
    )
    description_response = create_request_response(
        status_code=200,
        content_filepath=test_files_directory / "mock_responses" / "tasks" / "data_description_20.xml",
    )
    features_response = create_request_response(
        status_code=200,
        content_filepath=test_files_directory / "mock_responses" / "tasks" / "data_features_20.xml",
    )
    mock_get.side_effect = [task_response, description_response, features_response]

    task = openml.tasks.get_task(801)
    assert task.class_labels == ["tested_negative", "tested_positive"]
