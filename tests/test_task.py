import os

from openml.util import is_string
from openml.testing import TestBase
from openml import OpenMLSplit
import openml


class TestTask(TestBase):
    def test_list_tasks(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        def check_task(task):
            self.assertEqual(type(task), dict)
            self.assertGreaterEqual(len(task), 2)
            self.assertIn('did', task)
            self.assertIsInstance(task['did'], int)
            self.assertIn('status', task)
            self.assertTrue(is_string(task['status']))
            self.assertIn(task['status'],
                          ['in_preparation', 'active', 'deactivated'])

        # use a small task type as we cant limit tasks.
        # TODO inspect the tasks maybe?
        tasks = openml.tasks.list_tasks(self.connector, task_type_id=3)
        self.assertGreaterEqual(len(tasks), 300)
        for task in tasks:
            check_task(task)

    def test_get_task(self):
        task = openml.tasks.get_task(self.connector, 1)
        print(task)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "task.xml")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "datasets", "1", "dataset.arff")))

    def test_download_split(self):
        task = openml.tasks.get_task(self.connector, 1)
        split = task.download_split()
        self.assertEqual(type(split), OpenMLSplit)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))
