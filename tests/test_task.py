import os

from openml.util import is_string
from openml.testing import TestBase
from openml import OpenMLSplit
import openml


class TestTask(TestBase):
    def _check_task(self, task):
        self.assertEqual(type(task), dict)
        self.assertGreaterEqual(len(task), 2)
        self.assertIn('did', task)
        self.assertIsInstance(task['did'], int)
        self.assertIn('status', task)
        self.assertTrue(is_string(task['status']))
        self.assertIn(task['status'],
                      ['in_preparation', 'active', 'deactivated'])

    def test_list_tasks(self):
        tasks = openml.tasks.list_tasks()
        self.assertGreaterEqual(len(tasks), 2000)
        for task in tasks:
            self._check_task(task)

    def test_list_tasks_by_type(self):
        tasks = openml.tasks.list_tasks_by_type(task_type_id=3)
        self.assertGreaterEqual(len(tasks), 300)
        for task in tasks:
            self._check_task(task)

    def test_get_task(self):
        task = openml.tasks.get_task(1)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "task.xml")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "datasets", "1", "dataset.arff")))

    def test_download_split(self):
        task = openml.tasks.get_task(1)
        split = task.download_split()
        self.assertEqual(type(split), OpenMLSplit)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))
