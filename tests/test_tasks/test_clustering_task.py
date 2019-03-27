import openml
from openml.exceptions import OpenMLServerException

from tests.test_tasks import OpenMLTaskTest


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

    # overriding the method from the base
    # class. Ugly workaround but currently
    # there are no clustering tasks on the
    # test server. The task will be retrieved
    # from the main server and published on the
    # test server.
    def test_upload_task(self):

        task = openml.tasks.get_task(self.task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id)
        # No clustering tasks in the test server
        # TODO should be removed when issue is resolved
        openml.config.server = self.test_server
        # adding sentinel so we can have a new dataset
        # hence a "new task" to upload
        task.dataset_id = self._upload_dataset(dataset)
        task.estimation_procedure_id = self.estimation_procedure
        try:
            task.publish()
        except OpenMLServerException as e:
            # 614 is the error code
            # when the task already
            # exists
            if e.code != 614:
                raise e
