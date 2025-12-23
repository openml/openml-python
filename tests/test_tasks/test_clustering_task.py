# License: BSD 3-Clause
from __future__ import annotations

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

    @pytest.mark.production()
    def test_get_dataset(self):
        # no clustering tasks on test server
        self.use_production_server()
        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()

    @pytest.mark.production()
    def test_download_task(self):
        # no clustering tasks on test server
        self.use_production_server()
        task = super().test_download_task()
        assert task.task_id == self.task_id
        assert task.task_type_id == TaskType.CLUSTERING
        assert task.dataset_id == 36

    @pytest.mark.production()
    def test_estimation_procedure_extraction(self):
        # task 126033 has complete estimation procedure data
        self.use_production_server()
        task = openml.tasks.get_task(126033, download_data=False)

        assert task.task_type_id == TaskType.CLUSTERING
        assert task.estimation_procedure_id == 17

        est_proc = task.estimation_procedure
        assert est_proc["type"] == "testontrainingdata"
        assert est_proc["parameters"] is not None
        assert "number_repeats" in est_proc["parameters"]
        assert est_proc["data_splits_url"] is not None

    @pytest.mark.production()
    def test_estimation_procedure_empty_fields(self):
        # task 146714 has empty estimation procedure fields in XML
        self.use_production_server()
        task = openml.tasks.get_task(self.task_id, download_data=False)

        assert task.task_type_id == TaskType.CLUSTERING
        assert task.estimation_procedure_id == 17

    def test_upload_task(self):
        compatible_datasets = self._get_compatible_rand_dataset()
        for i in range(100):
            try:
                dataset_id = compatible_datasets[i % len(compatible_datasets)]
                # Upload a clustering task without a ground truth.
                task = openml.tasks.create_task(
                    task_type=self.task_type,
                    dataset_id=dataset_id,
                    estimation_procedure_id=self.estimation_procedure,
                )
                task = task.publish()
                TestBase._mark_entity_for_removal("task", task.id)
                TestBase.logger.info(
                    f"collected from {__file__.split('/')[-1]}: {task.id}",
                )
                # success
                break
            except OpenMLServerException as e:
                # Error code for 'task already exists'
                # Should be 533 according to the docs
                # (# https://www.openml.org/api_docs#!/task/post_task)
                if e.code == 614:
                    continue
                else:
                    raise e
        else:
            raise ValueError(
                f"Could not create a valid task for task type ID {self.task_type}",
            )
