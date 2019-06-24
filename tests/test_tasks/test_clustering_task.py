import openml
from .test_task import OpenMLTaskTest
from openml.exceptions import OpenMLServerException


class OpenMLClusteringTaskTest(OpenMLTaskTest):

    __test__ = True

    def setUp(self, n_levels: int = 1):

        super(OpenMLClusteringTaskTest, self).setUp()
        self.task_id = 146714
        self.task_type_id = 5
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
        self.assertEqual(task.task_type_id, 5)
        self.assertEqual(task.dataset_id, 36)

    def test_upload_task(self):

        # The base class uploads a clustering task with a target
        # feature. A situation where a ground truth is available
        # to benchmark the clustering algorithm.
        super(OpenMLClusteringTaskTest, self).test_upload_task()

        # Upload a clustering task without a ground truth.
        # As in the base class, we need to try different datasets
        # because the task may already exist.
        for i in range(100):
            try:
                dataset_id = self._get_compatible_rand_dataset()
                task = openml.tasks.create_task(
                    task_type_id=self.task_type_id,
                    dataset_id=dataset_id,
                    estimation_procedure_id=self.estimation_procedure
                )
                task_id = task.publish()

                # success
                break
            except OpenMLServerException as e:
                # Error code for 'task already exists'
                if e.code == 614:
                    continue
                else:
                    raise e
        else:
            raise ValueError(
                'Could not create a valid task for task type ID {}'.format(self.task_type_id)
            )

        openml.utils._delete_entity('task', task_id)
