import openml

from tests.test_tasks import OpenMLTaskTest


class OpenMLClusteringTest(OpenMLTaskTest):

    def setUp(self):

        self.task_id = 126101
        super(OpenMLClusteringTest, self).setUp()

    def test_get_dataset(self):

        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()
