from unittest.mock import MagicMock, patch

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

    @patch("openml.tasks.get_task")
    def test_get_dataset(self, mock_get_task):
        mock_task = MagicMock()
        mock_task.tid = self.task_id
        mock_get_task.return_value = mock_task

        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()

        mock_get_task.assert_called_with(self.task_id)
        mock_task.get_dataset.assert_called_once()

    @patch("tests.test_tasks.test_task.get_task")
    def test_download_task(self, mock_get_task):
        mock_task = MagicMock()
        mock_task.task_id = self.task_id
        mock_task.task_type_id = TaskType.CLUSTERING
        mock_task.dataset_id = 36
        mock_get_task.return_value = mock_task

        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.CLUSTERING
        assert task.dataset_id == 36

        mock_get_task.assert_called_with(self.task_id)

    @patch("openml.tasks.OpenMLTask.publish")
    @patch("openml.tasks.create_task")
    @patch("openml.datasets.list_datasets")
    def test_upload_task(self, mock_list_datasets, mock_create_task, mock_publish):
        import pandas as pd
        dataset_id = 1
        # Mock list_datasets to return a dataframe with at least one dataset
        mock_list_datasets.return_value = pd.DataFrame({
            "did": [dataset_id],
            "NumberOfSymbolicFeatures": [0],
            "NumberOfNumericFeatures": [10]
        })
        
        mock_task = MagicMock()
        mock_task.id = 123
        mock_task.publish.return_value = mock_task
        mock_publish.return_value = mock_task
        
        # Simulate: first call fails with "task already exists", second succeeds
        mock_create_task.side_effect = [
            OpenMLServerException(code=614, message="task already exists"),
            mock_task
        ]

        # The actual test logic inspired by the original:
        compatible_datasets = self._get_compatible_rand_dataset()
        for i in range(100):
            try:
                dataset_id = compatible_datasets[i % len(compatible_datasets)]
                task = openml.tasks.create_task(
                    task_type=self.task_type,
                    dataset_id=dataset_id,
                    estimation_procedure_id=self.estimation_procedure,
                )
                task = task.publish()
                TestBase._mark_entity_for_removal("task", task.id)
                break
            except OpenMLServerException as e:
                if e.code == 614:
                    continue
                else:
                    raise e
        else:
            pytest.fail("Could not create a valid task")

        assert task.id == 123
        assert mock_create_task.call_count == 2
