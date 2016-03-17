import inspect
import os
import sys
import unittest

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import numpy as np

from openml import OpenMLDataset
from openml import OpenMLSplit
from openml import OpenMLTask

"""
class OpenMLTaskTest(unittest.TestCase):
    @mock.patch.object(APIConnector, "__init__", autospec=True)
    def setUp(self, api_connector_mock):
        __file__ = inspect.getfile(OpenMLTaskTest)
        self.directory = os.path.dirname(__file__)
        self.split_filename = os.path.join(self.directory, "..",
                                           "files", "tasks", "datasplits.arff")

        api_connector_mock.return_value = None
        self.api_connector = APIConnector()
        self.task = OpenMLTask(1, "supervised classification", 1, "class",
                         "crossvalidation wth holdout", None, None, None,
                         None, self.api_connector)

    @unittest.skip("Does not work right now")
    @mock.patch.object(APIConnector, "get_dataset", autospec=True)
    def test_get_dataset(self, api_connector_mock):
        api_connector_mock.return_value = "Some strange string"
        retval = self.task.get_dataset()
        self.assertEqual(api_connector_mock.return_value, retval)
        api_connector_mock.assert_called_with(self.api_connector, 1)

    @unittest.skip("Does not work right now")
    @mock.patch.object(OpenMLTask, "get_dataset", autospec=True)
    def test_get_X_and_Y(self, task_mock):
        dataset = mock.create_autospec(OpenMLTask)
        dataset.get_pandas = lambda target=None: (pd.DataFrame(np.zeros((10, 10))),
                                                   pd.Series(np.zeros((10, ))))
        task_mock.return_value = dataset
        rval = self.dataset.get_X_and_Y()
        X, Y = self.task.get_X_and_Y()
        self.assertEqual((10, 10), X.shape)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertEqual((10, ), Y.shape)
        self.assertIsInstance(Y, pd.Series)

    @unittest.skip("Does not work right now")
    @mock.patch.object(APIConnector, "download_split", autospec=True)
    def test_get_train_and_test_split_indices(self, api_connector_mock):
        split = OpenMLSplit.from_arff_file(self.split_filename)
        api_connector_mock.return_value = split
        train_indices, test_indices = self.task.get_train_test_split_indices(
            0, 0)
        self.assertEqual(48, train_indices[0])
        self.assertEqual(46, train_indices[-1])
        self.assertEqual(73, test_indices[0])
        self.assertEqual(63, test_indices[-1])
        train_indices, test_indices = self.task.get_train_test_split_indices(
            2, 2)
        self.assertEqual(44, train_indices[0])
        self.assertEqual(46, train_indices[-1])
        self.assertEqual(19, test_indices[0])
        self.assertEqual(5, test_indices[-1])
        self.assertRaisesRegexp(ValueError, "Fold 10 not known",
                                self.task.get_train_test_split_indices, 10, 0)
        self.assertRaisesRegexp(ValueError, "Repeat 10 not known",
                                self.task.get_train_test_split_indices, 0, 10)

    def test_get_fold(self):
        X = np.arange(20)
        Y = np.array(([0] * 10) + ([1] * 10))
        splits = self.task.get_CV_fold(X, Y, 0, 2, shuffle=False)
        self.assertTrue(all(splits[0] ==
                            np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])))
        self.assertTrue(all(splits[1] ==
                             np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])))

        splits = self.task.get_CV_fold(X, Y, 0, 2, shuffle=True)
        self.assertEqual(5, sum(Y[splits[0]]))
        self.assertEqual(5, sum(Y[splits[1]]))
        self.assertTrue(all(splits[0] ==
                            np.array([13, 2, 9, 19, 4, 12, 7, 10, 14, 6])))
        self.assertTrue(all(splits[1] ==
                            np.array([0, 17, 15, 1, 8, 5, 11, 3, 18, 16])))
    """


