# License: BSD 3-Clause
from __future__ import annotations

import inspect
import os
from pathlib import Path

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
        self.arff_filepath = (
            Path(self.directory).parent
            / "files"
            / "org"
            / "openml"
            / "test"
            / "tasks"
            / "1882"
            / "datasplits.arff"
        )
        self.pd_filename = self.arff_filepath.with_suffix(".pkl.py3")

    def tearDown(self):
        try:
            os.remove(self.pd_filename)
        except (OSError, FileNotFoundError):
            #  Replaced bare except. Not sure why these exceptions are acceptable.
            pass

    def test_eq(self):
        split = OpenMLSplit._from_arff_file(self.arff_filepath)
        assert split == split

        split2 = OpenMLSplit._from_arff_file(self.arff_filepath)
        split2.name = "a"
        assert split != split2

        split2 = OpenMLSplit._from_arff_file(self.arff_filepath)
        split2.description = "a"
        assert split != split2

        split2 = OpenMLSplit._from_arff_file(self.arff_filepath)
        split2.split[10] = {}
        assert split != split2

        split2 = OpenMLSplit._from_arff_file(self.arff_filepath)
        split2.split[0][10] = {}
        assert split != split2

    def test_from_arff_file(self):
        split = OpenMLSplit._from_arff_file(self.arff_filepath)
        assert isinstance(split.split, dict)
        assert isinstance(split.split[0], dict)
        assert isinstance(split.split[0][0], dict)
        assert isinstance(split.split[0][0][0][0], np.ndarray)
        assert isinstance(split.split[0][0][0].train, np.ndarray)
        assert isinstance(split.split[0][0][0].train, np.ndarray)
        assert isinstance(split.split[0][0][0][1], np.ndarray)
        assert isinstance(split.split[0][0][0].test, np.ndarray)
        assert isinstance(split.split[0][0][0].test, np.ndarray)
        for i in range(10):
            for j in range(10):
                assert split.split[i][j][0].train.shape[0] >= 808
                assert split.split[i][j][0].test.shape[0] >= 89
                assert (
                    split.split[i][j][0].train.shape[0] + split.split[i][j][0].test.shape[0] == 898
                )

    def test_get_split(self):
        split = OpenMLSplit._from_arff_file(self.arff_filepath)
        train_split, test_split = split.get(fold=5, repeat=2)
        assert train_split.shape[0] == 808
        assert test_split.shape[0] == 90
        self.assertRaisesRegex(
            ValueError,
            "Repeat 10 not known",
            split.get,
            10,
            2,
        )
        self.assertRaisesRegex(
            ValueError,
            "Fold 10 not known",
            split.get,
            2,
            10,
        )
