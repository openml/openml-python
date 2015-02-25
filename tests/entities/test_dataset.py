import inspect
import unittest
import os

import numpy as np
import pandas as pd

from openml.entities.dataset import OpenMLDataset

class OpenMLDatasetTest(unittest.TestCase):
    def setUp(self):
        # Load dataset id 1
        __file__ = inspect.getfile(OpenMLDatasetTest)
        self.directory = os.path.dirname(__file__)
        self.arff_filename = os.path.join(self.directory, "..",
                                          "files", "dataset.arff")
        self.pandas_filename = os.path.join(self.directory, "..",
                                            "files", "dataset.pd")
        self.dataset = OpenMLDataset(1, "anneal", 1, "Lorem ipsum.",
                                     "arff", None, None, None,
                                     "2014-04-06 23:19:24", None, "Public",
                                     "http://openml.liacs.nl/files/download/2/dataset_2_anneal.ORIG.arff",
                                     "class", None, None, None, None,
                                     None, None, None, None, None,
                                     "939966a711925e333bf4aaadeaa71135",
                                     data_file=self.arff_filename)

    def tearDown(self):
        for file_ in [self.pandas_filename]:
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

    def test_get_pandas(self):
        # Basic usage
        rval, categorical = self.dataset.get_pandas()
        self.assertIsInstance(rval, pd.DataFrame)
        self.assertEqual(rval.values.dtype, np.float64)
        self.assertEqual((898, 39), rval.shape)
        self.assertEqual(len(categorical), 39)

    def test_get_pandas_with_target(self):
        X, y, categorical = self.dataset.get_pandas(target="class")
        self.assertEqual(X.values.dtype, np.float64)
        self.assertEqual(y.values.dtype, np.int64)
        self.assertEqual(X.shape, (898, 38))
        self.assertEqual(len(categorical), 38)
        self.assertNotIn("class", X)
        self.assertEqual(y.shape, (898, ))
        self.assertEqual(y.name, "class")

    def test_get_pandas_with_rowid(self):
        self.dataset.row_id_attribute = "condition"
        rval, categorical = self.dataset.get_pandas(include_row_id=True)
        self.assertEqual(rval.values.dtype, np.float64)
        self.assertEqual(rval.shape, (898, 39))
        self.assertEqual(len(categorical), 39)
        self.assertIn("condition", rval)
        rval, categorical = self.dataset.get_pandas(include_row_id=False)
        self.assertEqual(rval.values.dtype, np.float64)
        self.assertEqual(rval.shape, (898, 38))
        self.assertEqual(len(categorical), 38)
        self.assertNotIn("condition", rval)

        # TODO this is not yet supported!
        #rowid = ["condition", "formability"]
        #self.dataset.row_id_attribute = rowid
        #rval = self.dataset.get_pandas(include_row_id=False)

    def test_get_pandas_with_ignore_attributes(self):
        self.dataset.ignore_attributes = "condition"
        rval, categorical = self.dataset.get_pandas(include_ignore_attributes=True)
        self.assertEqual(rval.values.dtype, np.float64)
        self.assertEqual(rval.shape, (898, 39))
        self.assertEqual(len(categorical), 39)
        self.assertIn("condition", rval)
        rval, categorical = self.dataset.get_pandas(include_ignore_attributes=False)
        self.assertEqual(rval.values.dtype, np.float64)
        self.assertEqual(rval.shape, (898, 38))
        self.assertEqual(len(categorical), 38)
        self.assertNotIn("condition", rval)
        # TODO test multiple ignore attributes!

    def test_get_pandas_rowid_and_ignore(self):
        self.dataset.ignore_attributes = "condition"
        self.dataset.row_id_attribute = "condition"
        rval, categorical = self.dataset.get_pandas(include_ignore_attributes=False,
                                       include_row_id=False)
        self.assertEqual(rval.values.dtype, np.float64)
        self.assertEqual(rval.shape, (898, 38))
        self.assertEqual(len(categorical), 38)
        self.dataset.ignore_attributes = "hardness"
        rval, categorical = self.dataset.get_pandas(include_ignore_attributes=False,
                                       include_row_id=False)
        self.assertEqual(rval.values.dtype, np.float64)
        self.assertEqual(rval.shape, (898, 37))
        self.assertEqual(len(categorical), 37)

    def test_get_pandas_rowid_and_ignore_and_target(self):
        self.dataset.ignore_attributes = "condition"
        self.dataset.row_id_attribute = "hardness"
        X, y, categorical = self.dataset.get_pandas(target="class",
                                                 include_row_id=False,
                                       include_ignore_attributes=False)
        self.assertEqual(X.values.dtype, np.float64)
        self.assertEqual(y.values.dtype, np.int64)
        self.assertEqual(X.shape, (898, 36))
        self.assertEqual(len(categorical), 36)
        self.assertListEqual(categorical, [True]*3 + [False] + [True]*2 + [
            False] + [True]*23 + [False]*3 + [True]*3)
        self.assertEqual(y.shape, (898, ))