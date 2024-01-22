# License: BSD 3-Clause
from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing_extensions import NamedTuple

import arff  # type: ignore
import numpy as np


class Split(NamedTuple):
    """A single split of a dataset."""

    train: np.ndarray
    test: np.ndarray


class OpenMLSplit:
    """OpenML Split object.

    Parameters
    ----------
    name : int or str
    description : str
    split : dict
    """

    def __init__(
        self,
        name: int | str,
        description: str,
        split: dict[int, dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]],
    ):
        self.description = description
        self.name = name
        self.split: dict[int, dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}

        # Add splits according to repetition
        for repetition in split:
            _rep = int(repetition)
            self.split[_rep] = OrderedDict()
            for fold in split[_rep]:
                self.split[_rep][fold] = OrderedDict()
                for sample in split[_rep][fold]:
                    self.split[_rep][fold][sample] = split[_rep][fold][sample]

        self.repeats = len(self.split)

        # TODO(eddiebergman): Better error message
        if any(len(self.split[0]) != len(self.split[i]) for i in range(self.repeats)):
            raise ValueError("")

        self.folds = len(self.split[0])
        self.samples = len(self.split[0][0])

    def __eq__(self, other: Any) -> bool:
        if (
            (not isinstance(self, type(other)))
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
    def _from_arff_file(cls, filename: Path) -> OpenMLSplit:  # noqa: C901, PLR0912
        repetitions = None
        name = None

        pkl_filename = filename.with_suffix(".pkl.py3")

        if pkl_filename.exists():
            with pkl_filename.open("rb") as fh:
                # TODO(eddiebergman): Would be good to figure out what _split is and assert it is
                _split = pickle.load(fh)  # noqa: S301
            repetitions = _split["repetitions"]
            name = _split["name"]

        # Cache miss
        if repetitions is None:
            # Faster than liac-arff and sufficient in this situation!
            if not filename.exists():
                raise FileNotFoundError(f"Split arff {filename} does not exist!")

            file_data = arff.load(filename.open("r"), return_type=arff.DENSE_GEN)
            splits = file_data["data"]
            name = file_data["relation"]
            attrnames = [attr[0] for attr in file_data["attributes"]]

            repetitions = OrderedDict()

            type_idx = attrnames.index("type")
            rowid_idx = attrnames.index("rowid")
            repeat_idx = attrnames.index("repeat")
            fold_idx = attrnames.index("fold")
            sample_idx = attrnames.index("sample") if "sample" in attrnames else None

            for line in splits:
                # A line looks like type, rowid, repeat, fold
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
                    raise ValueError(type_)

            for repetition in repetitions:
                for fold in repetitions[repetition]:
                    for sample in repetitions[repetition][fold]:
                        repetitions[repetition][fold][sample] = Split(
                            np.array(repetitions[repetition][fold][sample][0], dtype=np.int32),
                            np.array(repetitions[repetition][fold][sample][1], dtype=np.int32),
                        )

            with pkl_filename.open("wb") as fh:
                pickle.dump({"name": name, "repetitions": repetitions}, fh, protocol=2)

        assert name is not None
        return cls(name, "", repetitions)

    def get(self, repeat: int = 0, fold: int = 0, sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Returns the specified data split from the CrossValidationSplit object.

        Parameters
        ----------
        repeat : int
            Index of the repeat to retrieve.
        fold : int
            Index of the fold to retrieve.
        sample : int
            Index of the sample to retrieve.

        Returns
        -------
        numpy.ndarray
            The data split for the specified repeat, fold, and sample.

        Raises
        ------
        ValueError
            If the specified repeat, fold, or sample is not known.
        """
        if repeat not in self.split:
            raise ValueError("Repeat %s not known" % str(repeat))
        if fold not in self.split[repeat]:
            raise ValueError("Fold %s not known" % str(fold))
        if sample not in self.split[repeat][fold]:
            raise ValueError("Sample %s not known" % str(sample))
        return self.split[repeat][fold][sample]
