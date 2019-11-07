# License: BSD 3-Clause

from collections import namedtuple, OrderedDict
import os
import pickle

import numpy as np
import arff


Split = namedtuple("Split", ["train", "test"])


class OpenMLSplit(object):
    """OpenML Split object.

       Parameters
       ----------
       name : int or str
       description : str
       split : dict
    """

    def __init__(self, name, description, split):
        self.description = description
        self.name = name
        self.split = dict()

        # Add splits according to repetition
        for repetition in split:
            repetition = int(repetition)
            self.split[repetition] = OrderedDict()
            for fold in split[repetition]:
                self.split[repetition][fold] = OrderedDict()
                for sample in split[repetition][fold]:
                    self.split[repetition][fold][sample] = split[
                        repetition][fold][sample]

        self.repeats = len(self.split)
        if any([len(self.split[0]) != len(self.split[i])
                for i in range(self.repeats)]):
            raise ValueError('')
        self.folds = len(self.split[0])
        self.samples = len(self.split[0][0])

    def __eq__(self, other):
        if (type(self) != type(other)
                or self.name != other.name
                or self.description != other.description
                or self.split.keys() != other.split.keys()):
            return False

        if any(self.split[repetition].keys() != other.split[repetition].keys()
                for repetition in self.split):
            return False

        samples = [(repetition, fold, sample)
                   for repetition in self.split
                   for fold in self.split[repetition]
                   for sample in self.split[repetition][fold]]

        for repetition, fold, sample in samples:
            self_train, self_test = self.split[repetition][fold][sample]
            other_train, other_test = other.split[repetition][fold][sample]
            if not (np.all(self_train == other_train)
                    and np.all(self_test == other_test)):
                return False
        return True

    @classmethod
    def _from_arff_file(cls, filename: str) -> 'OpenMLSplit':

        repetitions = None

        pkl_filename = filename.replace(".arff", ".pkl.py3")

        if os.path.exists(pkl_filename):
            with open(pkl_filename, "rb") as fh:
                _ = pickle.load(fh)
            repetitions = _["repetitions"]
            name = _["name"]

        # Cache miss
        if repetitions is None:
            # Faster than liac-arff and sufficient in this situation!
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    'Split arff %s does not exist!' % filename
                )
            file_data = arff.load(open(filename), return_type=arff.DENSE_GEN)
            splits = file_data['data']
            name = file_data['relation']
            attrnames = [attr[0] for attr in file_data['attributes']]

            repetitions = OrderedDict()

            type_idx = attrnames.index('type')
            rowid_idx = attrnames.index('rowid')
            repeat_idx = attrnames.index('repeat')
            fold_idx = attrnames.index('fold')
            sample_idx = (
                attrnames.index('sample')
                if 'sample' in attrnames
                else None
            )

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
                if type_ == 'TRAIN':
                    split[0].append(line[rowid_idx])
                elif type_ == 'TEST':
                    split[1].append(line[rowid_idx])
                else:
                    raise ValueError(type_)

            for repetition in repetitions:
                for fold in repetitions[repetition]:
                    for sample in repetitions[repetition][fold]:
                        repetitions[repetition][fold][sample] = Split(
                            np.array(repetitions[repetition][fold][sample][0],
                                     dtype=np.int32),
                            np.array(repetitions[repetition][fold][sample][1],
                                     dtype=np.int32))

            with open(pkl_filename, "wb") as fh:
                pickle.dump({"name": name, "repetitions": repetitions}, fh,
                            protocol=2)

        return cls(name, '', repetitions)

    def from_dataset(self, X, Y, folds, repeats):
        raise NotImplementedError()

    def get(self, repeat=0, fold=0, sample=0):
        if repeat not in self.split:
            raise ValueError("Repeat %s not known" % str(repeat))
        if fold not in self.split[repeat]:
            raise ValueError("Fold %s not known" % str(fold))
        if sample not in self.split[repeat][fold]:
            raise ValueError("Sample %s not known" % str(sample))
        return self.split[repeat][fold][sample]
