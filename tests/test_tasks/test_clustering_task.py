# License: BSD 3-Clause
from __future__ import annotations

from unittest import mock

import pytest

import openml
from openml.exceptions import OpenMLServerException
from openml.tasks import TaskType
from openml.testing import TestBase

from .test_task import OpenMLTaskTest


class OpenMLClusteringTaskTest(OpenMLTaskTest):
    __test__ = True

    def setUp(self, n_levels: int = 1):
        super().setUp()
        self.task_id = 146714
        self.task_type = TaskType.CLUSTERING
        self.estimation_procedure = 17

    @mock.patch("openml.datasets.get_dataset")
    def test_get_dataset(self, mock_get_dataset):
        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()
        mock_get_dataset.assert_called_once_with(task.dataset_id)


    @mock.patch("tests.test_tasks.test_task.get_task")
    def test_download_task(self, mock_get_task):
        mock_task = mock.Mock()
        mock_task.task_id = self.task_id
        mock_task.task_type_id = TaskType.CLUSTERING
        mock_task.dataset_id = 36
        mock_get_task.return_value = mock_task

        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.CLUSTERING
        assert task.dataset_id == 36
        mock_get_task.assert_called_with(self.task_id)

    @mock.patch("openml.tasks.create_task")
    def test_upload_task(self, mock_create_task):
        mock_task = mock.Mock()
        mock_task.publish.return_value = mock_task
        mock_task.id = 1
        mock_create_task.return_value = mock_task

        # Triggering the test
        task = openml.tasks.create_task(
            task_type=self.task_type,
            dataset_id=36,
            estimation_procedure_id=self.estimation_procedure,
        )
        task = task.publish()

        assert task.id == 1
        mock_create_task.assert_called_once_with(
            task_type=self.task_type,
            dataset_id=36,
            estimation_procedure_id=self.estimation_procedure,
        )
        mock_task.publish.assert_called_once()
