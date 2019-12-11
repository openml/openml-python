# License: BSD 3-Clause

import inspect
import os

import numpy as np

from openml import OpenMLSplit
from openml.testing import TestBase


class OpenMLSplitTest(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take less
    # than 5 seconds + rebuilding the test would potentially be costly

    def setUp(self):
        __file__ = inspect.getfile(OpenMLSplitTest)
        self.directory = os.path.dirname(__file__)
        # This is for dataset
        self.arff_filename = os.path.join(
            self.directory, "..", "files", "org", "openml", "test",
            "tasks", "1882", "datasplits.arff"
        )
        self.pd_filename = self.arff_filename.replace(".arff", ".pkl.py3")

    def tearDown(self):
        try:
            os.remove(self.pd_filename)
        except (OSError, FileNotFoundError):
            #  Replaced bare except. Not sure why these exceptions are acceptable.
            pass

    def test_eq(self):
        split = OpenMLSplit._from_arff_file(self.arff_filename)
        self.assertEqual(split, split)

        split2 = OpenMLSplit._from_arff_file(self.arff_filename)
        split2.name = "a"
        self.assertNotEqual(split, split2)

        split2 = OpenMLSplit._from_arff_file(self.arff_filename)
        split2.description = "a"
        self.assertNotEqual(split, split2)

        split2 = OpenMLSplit._from_arff_file(self.arff_filename)
        split2.split[10] = dict()
        self.assertNotEqual(split, split2)

        split2 = OpenMLSplit._from_arff_file(self.arff_filename)
        split2.split[0][10] = dict()
        self.assertNotEqual(split, split2)

    def test_from_arff_file(self):
        split = OpenMLSplit._from_arff_file(self.arff_filename)
        self.assertIsInstance(split.split, dict)
        self.assertIsInstance(split.split[0], dict)
        self.assertIsInstance(split.split[0][0], dict)
        self.assertIsInstance(split.split[0][0][0][0], np.ndarray)
        self.assertIsInstance(split.split[0][0][0].train, np.ndarray)
        self.assertIsInstance(split.split[0][0][0].train, np.ndarray)
        self.assertIsInstance(split.split[0][0][0][1], np.ndarray)
        self.assertIsInstance(split.split[0][0][0].test, np.ndarray)
        self.assertIsInstance(split.split[0][0][0].test, np.ndarray)
        for i in range(10):
            for j in range(10):
                self.assertGreaterEqual(split.split[i][j][0].train.shape[0], 808)
                self.assertGreaterEqual(split.split[i][j][0].test.shape[0], 89)
                self.assertEqual(split.split[i][j][0].train.shape[0]
                                 + split.split[i][j][0].test.shape[0],
                                 898)

    def test_get_split(self):
        split = OpenMLSplit._from_arff_file(self.arff_filename)
        train_split, test_split = split.get(fold=5, repeat=2)
        self.assertEqual(train_split.shape[0], 808)
        self.assertEqual(test_split.shape[0], 90)
        self.assertRaisesRegex(
            ValueError,
            "Repeat 10 not known",
            split.get,
            10, 2,
        )
        self.assertRaisesRegex(
            ValueError,
            "Fold 10 not known",
            split.get,
            2, 10,
        )
