import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

from time import time
import numpy as np

import openml
from openml.testing import TestBase


class OpenMLTaskTest(TestBase):
    _multiprocess_can_split_ = True

    @mock.patch('openml.tasks.functions.get_dataset', autospec=True)
    def test_get_dataset(self, patch):
        patch.return_value = mock.MagicMock()
        mm = mock.MagicMock()
        patch.return_value.retrieve_class_labels = mm
        patch.return_value.retrieve_class_labels.return_value = 'LA'
        retval = openml.tasks.get_task(1)
        self.assertEqual(patch.call_count, 1)
        self.assertIsInstance(retval, openml.OpenMLTask)
        self.assertEqual(retval.class_labels, 'LA')

    def test_get_X_and_Y(self):
        # Classification task
        task = openml.tasks.get_task(1)
        X, Y = task.get_X_and_y()
        self.assertEqual((898, 38), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((898, ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)

        # Regression task
        task = openml.tasks.get_task(631)
        X, Y = task.get_X_and_y()
        self.assertEqual((52, 2), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((52,), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, float)

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
