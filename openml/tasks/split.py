from collections import namedtuple, OrderedDict
import os
import six

import numpy as np
import scipy.io.arff
from six.moves import cPickle as pickle


Split = namedtuple("Split", ["train", "test"])


if six.PY2:
    FileNotFoundError = IOError


class OpenMLSplit(object):

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
                    self.split[repetition][fold][sample] = split[repetition][fold][sample]

        self.repeats = len(self.split)
        if any([len(self.split[0]) != len(self.split[i])
                for i in range(self.repeats)]):
            raise ValueError('')
        self.folds = len(self.split[0])
        self.samples = len(self.split[0][0])

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif self.name != other.name:
            return False
        elif self.description != other.description:
            return False
        elif self.split.keys() != other.split.keys():
            return False
        else:
            for repetition in self.split:
                if self.split[repetition].keys() != other.split[repetition].keys():
                    return False
                else:
                    for fold in self.split[repetition]:
                        for sample in self.split[repetition][fold]:
                            if np.all(self.split[repetition][fold][sample].test !=
                                      other.split[repetition][fold][sample].test)\
                                    and \
                                    np.all(self.split[repetition][fold][sample].train
                                           != other.split[repetition][fold][sample].train):
                                return False
        return True

    @classmethod
    def _from_arff_file(cls, filename, cache=True):
        repetitions = None
        if six.PY2:
            pkl_filename = filename.replace(".arff", ".pkl.py2")
        else:
            pkl_filename = filename.replace(".arff", ".pkl.py3")
        if cache:
            if os.path.exists(pkl_filename):
                try:
                    with open(pkl_filename, "rb") as fh:
                        _ = pickle.load(fh)
                except UnicodeDecodeError as e:
                    # Possibly pickle file was created with python2 and python3 is being used to load the data
                    raise e
                repetitions = _["repetitions"]
                name = _["name"]

        # Cache miss
        if repetitions is None:
            # Faster than liac-arff and sufficient in this situation!
            if not os.path.exists(filename):
                raise FileNotFoundError('Split arff %s does not exist!' % filename)
            splits, meta = scipy.io.arff.loadarff(filename)
            name = meta.name

            repetitions = OrderedDict()

            type_idx = meta._attrnames.index('type')
            rowid_idx = meta._attrnames.index('rowid')
            repeat_idx = meta._attrnames.index('repeat')
            fold_idx = meta._attrnames.index('fold')
            sample_idx = (meta._attrnames.index('sample') if 'sample' in meta._attrnames else None) # can be None

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

                type_ = line[type_idx].decode('utf-8')
                if type_ == 'TRAIN':
                    repetitions[repetition][fold][sample][0].append(line[rowid_idx])
                elif type_ == 'TEST':
                    repetitions[repetition][fold][sample][1].append(line[rowid_idx])
                else:
                    raise ValueError(type_)

            for repetition in repetitions:
                for fold in repetitions[repetition]:
                    for sample in repetitions[repetition][fold]:
                        repetitions[repetition][fold][sample] = Split(
                            np.array(repetitions[repetition][fold][sample][0], dtype=np.int32),
                            np.array(repetitions[repetition][fold][sample][1], dtype=np.int32))

            if cache:
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
