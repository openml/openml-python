import openml

from .test_task import OpenMLTaskTest


class OpenMLClusteringTaskTest(OpenMLTaskTest):

    def setUp(self):

        super(OpenMLClusteringTaskTest, self).setUp()
        # no clustering tasks on test server
        self.production_server = 'https://openml.org/api/v1/xml'
        self.test_server = 'https://test.openml.org/api/v1/xml'
        openml.config.server = self.production_server
        self.task_id = 126101
        self.estimation_procedure = 17

    def test_get_dataset(self):

        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()

    def test_download_task(self):

        task = super(OpenMLClusteringTaskTest, self).test_download_task()
        self.assertEqual(task.task_id, self.task_id)
        self.assertEqual(task.task_type_id, 5)
        self.assertEqual(task.dataset_id, 77)

    def test_upload_task(self):
        """
        Overrides test_upload_task from the base class.
        Ugly workaround but currently there are no clustering
        tasks on the test server. The task will be retrieved
        from the main server and published on the test server.
        """
        task = openml.tasks.get_task(self.task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
        # No clustering tasks in the test server
        # TODO should be removed when issue is resolved
        openml.config.server = self.test_server
        new_dataset_id = self._upload_dataset(dataset)
        OpenMLClusteringTaskTest._wait_dataset_activation(new_dataset_id, 10)
        task.dataset_id = new_dataset_id
        task.estimation_procedure_id = self.estimation_procedure
        task.publish()
