# License: BSD 3-Clause

from time import time

import openml
from openml.testing import TestBase


# Common methods between tasks
class OpenMLTaskMethodsTest(TestBase):

    def setUp(self):
        super(OpenMLTaskMethodsTest, self).setUp()

    def tearDown(self):
        super(OpenMLTaskMethodsTest, self).tearDown()

    def test_tagging(self):
        task = openml.tasks.get_task(1)
        tag = "testing_tag_{}_{}".format(self.id(), time())
        task_list = openml.tasks.list_tasks(tag=tag)
        self.assertEqual(len(task_list), 0)
        task.push_tag(tag)
        task_list = openml.tasks.list_tasks(tag=tag)
        self.assertEqual(len(task_list), 1)
        self.assertIn(1, task_list)
        task.remove_tag(tag)
        task_list = openml.tasks.list_tasks(tag=tag)
        self.assertEqual(len(task_list), 0)

    def test_get_train_and_test_split_indices(self):
        openml.config.cache_directory = self.static_cache_dir
        task = openml.tasks.get_task(1882)
        train_indices, test_indices = task.get_train_test_split_indices(0, 0)
        self.assertEqual(16, train_indices[0])
        self.assertEqual(395, train_indices[-1])
        self.assertEqual(412, test_indices[0])
        self.assertEqual(364, test_indices[-1])
        train_indices, test_indices = task.get_train_test_split_indices(2, 2)
        self.assertEqual(237, train_indices[0])
        self.assertEqual(681, train_indices[-1])
        self.assertEqual(583, test_indices[0])
        self.assertEqual(24, test_indices[-1])
        self.assertRaisesRegexp(ValueError, "Fold 10 not known",
                                task.get_train_test_split_indices, 10, 0)
        self.assertRaisesRegexp(ValueError, "Repeat 10 not known",
                                task.get_train_test_split_indices, 0, 10)
