import openml
from .test_task import OpenMLTaskTest


class OpenMLClusteringTaskTest(OpenMLTaskTest):

    def setUp(self, n_levels: int = 1):

        super(OpenMLClusteringTaskTest, self).setUp()
        self.task_id = 146714
        self.task_type_id = 5
        self.estimation_procedure = 17
        self.production_server = "https://openml.org/api/v1/xml"

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
