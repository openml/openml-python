import openml

from tests.test_tasks.test_task import OpenMLTaskTest


class OpenMLClusteringTest(OpenMLTaskTest):

    def setup(self):

        self.task_id = 126101

    def test_get_dataset(self):

        task = openml.tasks.get_task(self.task_id)
        task.get_dataset()
