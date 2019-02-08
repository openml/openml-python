import openml
from openml.exceptions import OpenMLServerException

from tests.test_tasks import OpenMLTaskTest


class OpenMLClusteringTest(OpenMLTaskTest):

    def setUp(self):

        super(OpenMLClusteringTest, self).setUp()
        # no clustering tasks on test server
        self.production_server = 'https://openml.org/api/v1/xml'
        self.test_server = 'https://test.openml.org/api/v1/xml'
        openml.config.server = self.production_server
        self.task_id = 126101

    def test_get_dataset(self):

        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()

    # overriding the method from the base
    # class. Ugly workaround but currently
    # there are no clustering tasks on the
    # test server. The task will be retrieved
    # from the main server and published on the
    # test server.
    def test_upload_task(self):

        task = openml.tasks.get_task(self.task_id)
        openml.config.server = self.test_server
        task.estimation_procedure_id = 23
        try:
            task.publish()
        except OpenMLServerException as e:
            # 614 is the error code
            # when the task already
            # exists
            if e.code != 614:
                raise e
