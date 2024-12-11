# License: BSD 3-Clause
from __future__ import annotations

import os
import pickle
from collections import OrderedDict, namedtuple

import arff
import numpy as np

# Named tuple to represent a train-test split
Split = namedtuple("Split", ["train", "test"])


class OpenMLSplit:
    """
    Represents a split object for OpenML datasets.

    This class manages train-test splits for a dataset across multiple
    repetitions, folds, and samples.

    Parameters
    ----------
    name : int or str
        The name or ID of the split.
    description : str
        A textual description of the split.
    split : dict
        A dictionary containing the splits organized by repetition, fold,
        and sample.

    Attributes
    ----------
    name : int or str
        The name or ID of the split.
    description : str
        Description of the split.
    split : dict
        Nested dictionary holding the train-test indices for each repetition,
        fold, and sample.
    repeats : int
        Number of repetitions in the split.
    folds : int
        Number of folds in each repetition.
    samples : int
        Number of samples in each fold.

    Raises
    ------
    ValueError
        If the number of folds is inconsistent across repetitions.
    """

    def __init__(self, name, description, split):
        self.description = description
        self.name = name
        self.split = {}

        # Populate splits according to repetitions
        for repetition in split:
            repetition = int(repetition)
            self.split[repetition] = OrderedDict()
            for fold in split[repetition]:
                self.split[repetition][fold] = OrderedDict()
                for sample in split[repetition][fold]:
                    self.split[repetition][fold][sample] = split[repetition][fold][sample]

        self.repeats = len(self.split)
        if any(len(self.split[0]) != len(self.split[i]) for i in range(self.repeats)):
            raise ValueError("Number of folds is inconsistent across repetitions.")
        self.folds = len(self.split[0])
        self.samples = len(self.split[0][0])

    def __eq__(self, other):
        """
        Check if two OpenMLSplit objects are equal.

        Parameters
        ----------
        other : OpenMLSplit
            Another OpenMLSplit object to compare against.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if (
            type(self) != type(other)
            or self.name != other.name
            or self.description != other.description
            or self.split.keys() != other.split.keys()
            or any(
                self.split[repetition].keys() != other.split[repetition].keys()
                for repetition in self.split
            )
        ):
            return False

        samples = [
            (repetition, fold, sample)
            for repetition in self.split
            for fold in self.split[repetition]
            for sample in self.split[repetition][fold]
        ]

        for repetition, fold, sample in samples:
            self_train, self_test = self.split[repetition][fold][sample]
            other_train, other_test = other.split[repetition][fold][sample]
            if not (np.all(self_train == other_train) and np.all(self_test == other_test)):
                return False
        return True

    @classmethod
    def _from_arff_file(cls, filename: str) -> OpenMLSplit:
        """
        Create an OpenMLSplit object from an ARFF file.

        Parameters
        ----------
        filename : str
            Path to the ARFF file.

        Returns
        -------
        OpenMLSplit
            The constructed OpenMLSplit object.

        Raises
        ------
        FileNotFoundError
            If the ARFF file does not exist.
        ValueError
            If an unknown split type is encountered.
        """
        repetitions = None
        pkl_filename = filename.replace(".arff", ".pkl.py3")

        # Try loading from a cached pickle file
        if os.path.exists(pkl_filename):
            with open(pkl_filename, "rb") as fh:
                _ = pickle.load(fh)
            repetitions = _["repetitions"]
            name = _["name"]

        # Cache miss: load from ARFF file
        if repetitions is None:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Split ARFF file {filename} does not exist.")
            file_data = arff.load(open(filename), return_type=arff.DENSE_GEN)
            splits = file_data["data"]
            name = file_data["relation"]
            attrnames = [attr[0] for attr in file_data["attributes"]]

            repetitions = OrderedDict()

            # Identify attribute indices
            type_idx = attrnames.index("type")
            rowid_idx = attrnames.index("rowid")
            repeat_idx = attrnames.index("repeat")
            fold_idx = attrnames.index("fold")
            sample_idx = attrnames.index("sample") if "sample" in attrnames else None

            for line in splits:
                repetition = int(line[repeat_idx])
                fold = int(line[fold_idx])
                sample = 0
                if sample_idx is not None:
                    sample = int(line[sample_idx])

                if repetition not in repetitions:
                    repetitions[repetition] = OrderedDict()
                if fold not in repetitions[repetition]:
                    repetitions[repetition][fold] = OrderedDict()
                if sample not in repetitions[repetition][fold]:
                    repetitions[repetition][fold][sample] = ([], [])
                split = repetitions[repetition][fold][sample]

                type_ = line[type_idx]
                if type_ == "TRAIN":
                    split[0].append(line[rowid_idx])
                elif type_ == "TEST":
                    split[1].append(line[rowid_idx])
                else:
                    raise ValueError(f"Unknown split type: {type_}")

            # Convert lists to numpy arrays
            for repetition in repetitions:
                for fold in repetitions[repetition]:
                    for sample in repetitions[repetition][fold]:
                        repetitions[repetition][fold][sample] = Split(
                            np.array(repetitions[repetition][fold][sample][0], dtype=np.int32),
                            np.array(repetitions[repetition][fold][sample][1], dtype=np.int32),
                        )

            # Cache the parsed splits
            with open(pkl_filename, "wb") as fh:
                pickle.dump({"name": name, "repetitions": repetitions}, fh, protocol=2)

        return cls(name, "", repetitions)

    def from_dataset(self, X, Y, folds, repeats):
        """
        Construct splits from a dataset.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        Y : array-like
            Target array.
        folds : int
            Number of folds.
        repeats : int
            Number of repetitions.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("from_dataset method is not implemented.")

    def get(self, repeat=0, fold=0, sample=0):
        """
        Retrieve a specific split.

        Parameters
        ----------
        repeat : int, optional
            Repetition index (default is 0).
        fold : int, optional
            Fold index (default is 0).
        sample : int, optional
            Sample index (default is 0).

        Returns
        -------
        Split
            A named tuple containing train and test indices.

        Raises
        ------
        ValueError
            If the specified repeat, fold, or sample does not exist.
        """
        if repeat not in self.split:
            raise ValueError(f"Repeat {repeat} not known.")
        if fold not in self.split[repeat]:
            raise ValueError(f"Fold {fold} not known.")
        if sample not in self.split[repeat][fold]:
            raise ValueError(f"Sample {sample} not known.")
        return self.split[repeat][fold][sample]
