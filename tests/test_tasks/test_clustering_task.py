import openml

from .test_task import OpenMLTaskTest


class OpenMLClusteringTaskTest(OpenMLTaskTest):

    def setUp(self):

        super(OpenMLClusteringTaskTest, self).setUp()
        # no clustering tasks on test server
        self.production_server = 'https://openml.org/api/v1/xml'
        self.test_server = 'https://test.openml.org/api/v1/xml'
        openml.config.server = self.production_server
        self.task_id = 146714
        self.task_type_id = 5
        self.estimation_procedure = 17
        self.dataset_id_test = 19

    def test_get_dataset(self):

        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()

    def test_download_task(self):

        task = super(OpenMLClusteringTaskTest, self).test_download_task()
        self.assertEqual(task.task_id, self.task_id)
        self.assertEqual(task.task_type_id, 5)
        self.assertEqual(task.dataset_id, 36)
