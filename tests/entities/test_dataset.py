import inspect
import unittest
import os

import numpy as np

from openml import OpenMLDataset
from openml.util import is_string


class OpenMLDatasetTest(unittest.TestCase):

    def setUp(self):
        # Load dataset id 1
        __file__ = inspect.getfile(OpenMLDatasetTest)
        self.directory = os.path.dirname(__file__)
        self.arff_filename = os.path.join(self.directory, "..", "files",
                                          "datasets", "2", "dataset.arff")
        self.pickle_filename = os.path.join(self.directory, "..", "files",
                                            "datasets", "2", "dataset.pkl")
        self.dataset = OpenMLDataset(
            1, "anneal", 2, "Lorem ipsum.", "arff", None, None, None,
            "2014-04-06 23:19:24", None, "Public",
            "http://openml.liacs.nl/files/download/2/dataset_2_anneal.ORIG.arff",
            "class", None, None, None, None, None, None, None, None, None,
            "939966a711925e333bf4aaadeaa71135", data_file=self.arff_filename)

        self.sparse_arff_filename = os.path.join(
            self.directory, "..", "files", "datasets", "-1", "dataset.arff")
        self.sparse_pickle_filename = os.path.join(
            self.directory, "..", "files", "datasets", "-1", "dataset.pkl")
        self.sparse_dataset = OpenMLDataset(
            -1, "dexter", -1, "Lorem ipsum.", "arff", None, None, None, None,
            None, "Public",
            "http://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets/dexter.zip",
            "class", None, None, None, None, None, None, None, None, None,
            None, data_file=self.sparse_arff_filename)

    def tearDown(self):
        for file_ in [self.pickle_filename, self.sparse_pickle_filename]:
            os.remove(file_)

    ##########################################################################
    # Pandas

    @unittest.skip("Does not work right now")
    def test_get_arff(self):
        rval = self.dataset.get_arff()
        self.assertIsInstance(rval, tuple)
        self.assertIsInstance(rval[0], np.ndarray)
        self.assertTrue(hasattr(rval[1], '__dict__'))
        self.assertEqual(rval[0].shape, (898, ))

    def test_get_data(self):
        # Basic usage
        rval = self.dataset.get_data()
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual((898, 39), rval.shape)
        rval, categorical = self.dataset.get_data(
            return_categorical_indicator=True)
        self.assertEqual(len(categorical), 39)
        self.assertTrue(all([isinstance(cat, bool) for cat in categorical]))
        rval, attribute_names = self.dataset.get_data(
            return_attribute_names=True)
        self.assertEqual(len(attribute_names), 39)
        self.assertTrue(all([is_string(att) for att in attribute_names]))

    def test_get_sparse_dataset(self):
        rval = self.sparse_dataset.get_data()
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual((2, 20001), rval.shape)
        rval, categorical = self.sparse_dataset.get_data(
            return_categorical_indicator=True)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(len(categorical), 20001)
        self.assertTrue(all([isinstance(cat, bool) for cat in categorical]))
        rval, attribute_names = self.sparse_dataset.get_data(
            return_attribute_names=True)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(len(attribute_names), 20001)
        self.assertTrue(all([is_string(att) for att in attribute_names]))

    def test_get_data_with_target(self):
        X, y = self.dataset.get_data(target="class")
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(X.shape, (898, 38))
        X, y, attribute_names = self.dataset.get_data(
            target="class", return_attribute_names=True)
        self.assertEqual(len(attribute_names), 38)
        self.assertNotIn("class", attribute_names)
        self.assertEqual(y.shape, (898, ))

    def test_get_sparse_dataset_with_target(self):
        X, y = self.sparse_dataset.get_data(target="class")
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.dtype, np.float32)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(X.shape, (2, 20000))
        X, y, attribute_names = self.sparse_dataset.get_data(
            target="class", return_attribute_names=True)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(len(attribute_names), 20000)
        self.assertNotIn("class", attribute_names)
        self.assertEqual(y.shape, (2, ))

    def test_get_data_with_rowid(self):
        self.dataset.row_id_attribute = "condition"
        rval, categorical = self.dataset.get_data(
            include_row_id=True, return_categorical_indicator=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 39))
        self.assertEqual(len(categorical), 39)
        rval, categorical = self.dataset.get_data(
            include_row_id=False, return_categorical_indicator=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 38))
        self.assertEqual(len(categorical), 38)

        # TODO this is not yet supported!
        #rowid = ["condition", "formability"]
        #self.dataset.row_id_attribute = rowid
        #rval = self.dataset.get_pandas(include_row_id=False)

    def test_get_sparse_dataset_with_rowid(self):
        self.sparse_dataset.row_id_attribute = "a_0"
        rval, categorical = self.sparse_dataset.get_data(
            include_row_id=True, return_categorical_indicator=True)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (2, 20001))
        self.assertEqual(len(categorical), 20001)
        rval, categorical = self.sparse_dataset.get_data(
            include_row_id=False, return_categorical_indicator=True)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (2, 20000))
        self.assertEqual(len(categorical), 20000)

        # TODO this is not yet supported!
        # rowid = ["condition", "formability"]
        #self.dataset.row_id_attribute = rowid
        #rval = self.dataset.get_pandas(include_row_id=False)

    def test_get_data_with_ignore_attributes(self):
        self.dataset.ignore_attributes = "condition"
        rval = self.dataset.get_data(include_ignore_attributes=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 39))
        rval, categorical = self.dataset.get_data(
            include_ignore_attributes=True, return_categorical_indicator=True)
        self.assertEqual(len(categorical), 39)
        rval = self.dataset.get_data(include_ignore_attributes=False)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 38))
        rval, categorical = self.dataset.get_data(
            include_ignore_attributes=False, return_categorical_indicator=True)
        self.assertEqual(len(categorical), 38)
        # TODO test multiple ignore attributes!

    def test_get_sparse_dataset_with_ignore_attributes(self):
        self.sparse_dataset.ignore_attributes = "a_0"
        rval = self.sparse_dataset.get_data(include_ignore_attributes=True)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (2, 20001))
        rval, categorical = self.sparse_dataset.get_data(
            include_ignore_attributes=True, return_categorical_indicator=True)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(len(categorical), 20001)
        rval = self.sparse_dataset.get_data(include_ignore_attributes=False)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (2, 20000))
        rval, categorical = self.sparse_dataset.get_data(
            include_ignore_attributes=False, return_categorical_indicator=True)
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(len(categorical), 20000)
        # TODO test multiple ignore attributes!

    def test_get_data_rowid_and_ignore_and_target(self):
        self.dataset.ignore_attributes = "condition"
        self.dataset.row_id_attribute = "hardness"
        X, y = self.dataset.get_data(target="class", include_row_id=False,
                                     include_ignore_attributes=False)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(X.shape, (898, 36))
        X, y, categorical = self.dataset.get_data(
            target="class", return_categorical_indicator=True)
        self.assertEqual(len(categorical), 36)
        self.assertListEqual(categorical, [True] * 3 + [False] + [True] * 2 + [
            False] + [True] * 23 + [False] * 3 + [True] * 3)
        self.assertEqual(y.shape, (898, ))

    def test_get_sparse_dataset_rowid_and_ignore_and_target(self):
        self.sparse_dataset.ignore_attributes = "a_0"
        self.sparse_dataset.row_id_attribute = "a_1"
        X, y = self.sparse_dataset.get_data(
            target="class", include_row_id=False,
            include_ignore_attributes=False)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(X.shape, (2, 19998))
        X, y, categorical = self.sparse_dataset.get_data(
            target="class", return_categorical_indicator=True)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(len(categorical), 19998)
        self.assertListEqual(categorical, [False] * 19998)
        self.assertEqual(y.shape, (2, ))
