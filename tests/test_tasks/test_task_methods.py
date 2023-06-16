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
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        tag = "testing_tag_{}_{}".format(self.id(), time())
        tasks = openml.tasks.list_tasks(tag=tag, output_format="dataframe")
        self.assertEqual(len(tasks), 0)
        task.push_tag(tag)
        tasks = openml.tasks.list_tasks(tag=tag, output_format="dataframe")
        self.assertEqual(len(tasks), 1)
        self.assertIn(1, tasks["tid"])
        task.remove_tag(tag)
        tasks = openml.tasks.list_tasks(tag=tag, output_format="dataframe")
        self.assertEqual(len(tasks), 0)

    def test_get_train_and_test_split_indices(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
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
        self.assertRaisesRegex(
            ValueError, "Fold 10 not known", task.get_train_test_split_indices, 10, 0
        )
        self.assertRaisesRegex(
            ValueError, "Repeat 10 not known", task.get_train_test_split_indices, 0, 10
        )
