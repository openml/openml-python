import inspect
import os
import unittest

import numpy as np

from openml.entities.split import OpenMLSplit, Split


class OpenMLSplitTest(unittest.TestCase):
    def setUp(self):
        __file__ = inspect.getfile(OpenMLSplitTest)
        self.directory = os.path.dirname(__file__)
        # This is for dataset
        self.arff_filename = os.path.join(self.directory, "..",
                                          "files", "tasks", "datasplits.arff")
        self.pd_filename = self.arff_filename.replace(".arff", ".pkl")

    def tearDown(self):
        try:
            os.remove(self.pd_filename)
        except:
            pass

    def test_eq(self):
        split = OpenMLSplit.from_arff_file(self.arff_filename)
        self.assertEqual(split, split)

        split2 = OpenMLSplit.from_arff_file(self.arff_filename)
        split2.name = "a"
        self.assertNotEqual(split, split2)

        split2 = OpenMLSplit.from_arff_file(self.arff_filename)
        split2.description = "a"
        self.assertNotEqual(split, split2)

        split2 = OpenMLSplit.from_arff_file(self.arff_filename)
        split2.split[10] = dict()
        self.assertNotEqual(split, split2)

        split2 = OpenMLSplit.from_arff_file(self.arff_filename)
        split2.split[0][10] = dict()
        self.assertNotEqual(split, split2)

        split2 = OpenMLSplit.from_arff_file(self.arff_filename)
        split2.split[0][0] = Split(np.zeros((80)), np.zeros((9)))
        self.assertNotEqual(split, split2)

    def test_from_arff_file(self):
        split = OpenMLSplit.from_arff_file(self.arff_filename)
        self.assertIsInstance(split.split, dict)
        self.assertIsInstance(split.split[0], dict)
        self.assertIsInstance(split.split[0][0], Split)
        self.assertIsInstance(split.split[0][0][0], np.ndarray)
        self.assertIsInstance(split.split[0][0].train, np.ndarray)
        self.assertIsInstance(split.split[0][0].train, np.ndarray)
        self.assertIsInstance(split.split[0][0][1], np.ndarray)
        self.assertIsInstance(split.split[0][0].test, np.ndarray)
        self.assertIsInstance(split.split[0][0].test, np.ndarray)
        for i in range(10):
            for j in range(10):
                self.assertEqual((81,), split.split[i][j].train.shape)
                self.assertEqual((9,), split.split[i][j].test.shape)

    def test_get_split(self):
        split = OpenMLSplit.from_arff_file(self.arff_filename)
        train_split, test_split = split.get(fold=5, repeat=2)
        self.assertEqual(train_split.shape, (81,))
        self.assertEqual(test_split.shape, (9,))
        self.assertRaisesRegexp(ValueError, "Repeat 10 not known",
                                split.get, 10, 2)
        self.assertRaisesRegexp(ValueError, "Fold 10 not known",
                                split.get, 2, 10)