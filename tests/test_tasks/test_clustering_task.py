# License: BSD 3-Clause

import openml
from openml.tasks import TaskType
from openml.testing import TestBase
from .test_task import OpenMLTaskTest
from openml.exceptions import OpenMLServerException


class OpenMLClusteringTaskTest(OpenMLTaskTest):

    __test__ = True

    def setUp(self, n_levels: int = 1):

        super(OpenMLClusteringTaskTest, self).setUp()
        self.task_id = 146714
        self.task_type = TaskType.CLUSTERING
        self.estimation_procedure = 17

    def test_get_dataset(self):
        # no clustering tasks on test server
        openml.config.server = self.production_server
        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()

    def test_download_task(self):
        # no clustering tasks on test server
        openml.config.server = self.production_server
        task = super(OpenMLClusteringTaskTest, self).test_download_task()
        self.assertEqual(task.task_id, self.task_id)
        self.assertEqual(task.task_type_id, TaskType.CLUSTERING)
        self.assertEqual(task.dataset_id, 36)

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
                    "collected from {}: {}".format(__file__.split("/")[-1], task.id)
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
                "Could not create a valid task for task type ID {}".format(self.task_type)
            )
