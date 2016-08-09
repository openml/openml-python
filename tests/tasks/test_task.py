import sys
import types

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import numpy as np

import openml
from openml.testing import TestBase


class OpenMLTaskTest(TestBase):

    def test_get_clustering_task(self):
        self.assertRaisesRegexp(KeyError, 'oml:target_feature',
                                openml.tasks.get_task, 10128)

    @mock.patch('openml.datasets.get_dataset')
    def test_get_dataset(self, patch):
        patch.return_value = mock.MagicMock()
        mm = mock.MagicMock()
        patch.return_value._retrieve_class_labels = mm
        patch.return_value._retrieve_class_labels.return_value = 'LA'
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
        task = openml.tasks.get_task(2280)
        X, Y = task.get_X_and_y()
        self.assertEqual((8192, 8), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((8192,), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, float)

    def test_get_train_and_test_split_indices(self):
        openml.config.set_cache_directory(self.static_cache_dir)
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

    def test_iterate_repeats(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1882)

        num_repeats = 0
        for rep in task.iterate_repeats():
            num_repeats += 1
            self.assertIsInstance(rep, types.GeneratorType)
        self.assertEqual(num_repeats, 10)

    def test_iterate_all_splits(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1882)

        num_splits = 0
        for split in task.iterate_all_splits():
            num_splits += 1
            self.assertIsInstance(split[0], np.ndarray)
            self.assertIsInstance(split[1], np.ndarray)
            self.assertEqual(split[0].shape[0] + split[1].shape[0], 898)
        self.assertEqual(num_splits, 100)


