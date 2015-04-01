import inspect
import unittest
import os

import numpy as np
import pandas as pd

from openml.entities.dataset import OpenMLDataset
from openml.util import is_string

class OpenMLDatasetTest(unittest.TestCase):
    def setUp(self):
        # Load dataset id 1
        __file__ = inspect.getfile(OpenMLDatasetTest)
        self.directory = os.path.dirname(__file__)
        self.arff_filename = os.path.join(self.directory, "..",
            "files", "datasets", "2", "dataset.arff")
        self.pickle_filename = os.path.join(self.directory, "..",
            "files", "datasets", "2", "dataset.pkl")
        self.dataset = OpenMLDataset(1, "anneal", 1, "Lorem ipsum.",
                                     "arff", None, None, None,
                                     "2014-04-06 23:19:24", None, "Public",
                                     "http://openml.liacs.nl/files/download/2/dataset_2_anneal.ORIG.arff",
                                     "class", None, None, None, None,
                                     None, None, None, None, None,
                                     "939966a711925e333bf4aaadeaa71135",
                                     data_file=self.arff_filename)

    def tearDown(self):
        for file_ in [self.pickle_filename]:
            os.remove(file_)

    ############################################################################
    # Pandas

    @unittest.skip("Does not work right now")
    def test_get_arff(self):
        rval = self.dataset.get_arff()
        self.assertIsInstance(rval, tuple)
        self.assertIsInstance(rval[0], pd.DataFrame)
        self.assertTrue(hasattr(rval[1], '__dict__'))
        self.assertEqual(rval[0].shape, (898, ))

    def test_get_dataset(self):
        # Basic usage
        rval = self.dataset.get_dataset()
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual((898, 39), rval.shape)
        rval, categorical = self.dataset.get_dataset(
            return_categorical_indicator=True)
        self.assertEqual(len(categorical), 39)
        self.assertTrue(all([isinstance(cat, bool) for cat in categorical]))
        rval, attribute_names = self.dataset.get_dataset(
            return_attribute_names=True)
        self.assertEqual(len(attribute_names), 39)
        self.assertTrue(all([is_string(att) for att in attribute_names]))

    def test_get_dataset_with_target(self):
        X, y = self.dataset.get_dataset(target="class")
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int32)
        self.assertEqual(X.shape, (898, 38))
        X, y, attribute_names = self.dataset.get_dataset(
            target="class", return_attribute_names=True)
        self.assertEqual(len(attribute_names), 38)
        self.assertNotIn("class", attribute_names)
        self.assertEqual(y.shape, (898, ))

    def test_get_dataset_with_rowid(self):
        self.dataset.row_id_attribute = "condition"
        rval, categorical = self.dataset.get_dataset(
            include_row_id=True, return_categorical_indicator=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 39))
        self.assertEqual(len(categorical), 39)
        rval, categorical = self.dataset.get_dataset(
            include_row_id=False, return_categorical_indicator=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 38))
        self.assertEqual(len(categorical), 38)

        # TODO this is not yet supported!
        #rowid = ["condition", "formability"]
        #self.dataset.row_id_attribute = rowid
        #rval = self.dataset.get_pandas(include_row_id=False)

    def test_get_dataset_with_ignore_attributes(self):
        self.dataset.ignore_attributes = "condition"
        rval = self.dataset.get_dataset(include_ignore_attributes=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 39))
        rval, categorical = self.dataset.get_dataset(
            include_ignore_attributes=True, return_categorical_indicator=True)
        self.assertEqual(len(categorical), 39)
        rval = self.dataset.get_dataset(include_ignore_attributes=False)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 38))
        rval, categorical = self.dataset.get_dataset(
            include_ignore_attributes=False, return_categorical_indicator=True)
        self.assertEqual(len(categorical), 38)
        # TODO test multiple ignore attributes!

    def test_get_dataset_rowid_and_ignore(self):
        self.dataset.ignore_attributes = "condition"
        self.dataset.row_id_attribute = "condition"
        rval = self.dataset.get_dataset(include_ignore_attributes=False,
                                        include_row_id=False)
        self.assertEqual(rval.dtype, np.float32)

    def test_get_dataset_rowid_and_ignore_and_target(self):
        self.dataset.ignore_attributes = "condition"
        self.dataset.row_id_attribute = "hardness"
        X, y = self.dataset.get_dataset(target="class", include_row_id=False,
                                        include_ignore_attributes=False)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int32)
        self.assertEqual(X.shape, (898, 36))
        X, y , categorical = self.dataset.get_dataset(
            target="class", return_categorical_indicator=True)
        self.assertEqual(len(categorical), 36)
        self.assertListEqual(categorical, [True]*3 + [False] + [True]*2 + [
            False] + [True]*23 + [False]*3 + [True]*3)
        self.assertEqual(y.shape, (898, ))