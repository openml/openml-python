from collections import namedtuple, OrderedDict
import os
import sys
import numpy as np
import scipy.io.arff

if sys.version_info[0] > 3:
    import pickle
else:
    try:
        import cPickle as pickle
    except:
        import pickle


Split = namedtuple("Split", ["train", "test"])


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
                self.split[repetition][fold] = split[repetition][fold]

        self.repeats = len(self.split)
        if any([len(self.split[0]) != len(self.split[i])
                for i in range(self.repeats)]):
            raise ValueError('')
        self.folds = len(self.split[0])

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
                        if np.all(self.split[repetition][fold].test !=
                                  other.split[repetition][fold].test)\
                                and \
                                np.all(self.split[repetition][fold].train
                                       != other.split[repetition][fold].train):
                            return False
        return True

    @classmethod
    def from_arff_file(cls, filename, cache=True):
        repetitions = None
        pkl_filename = filename.replace(".arff", ".pkl")
        if cache:
            if os.path.exists(pkl_filename):
                with open(pkl_filename, "rb") as fh:
                    _ = pickle.load(fh)
                repetitions = _["repetitions"]
                name = _["name"]

        # Cache miss
        if repetitions is None:
            # Faster than liac-arff and sufficient in this situation!
            splits, meta = scipy.io.arff.loadarff(filename)
            name = meta.name

            repetitions = OrderedDict()
            for line in splits:
                # A line looks like type, rowid, repeat, fold
                repetition = int(line[2])
                fold = int(line[3])

                if repetition not in repetitions:
                    repetitions[repetition] = OrderedDict()
                if fold not in repetitions[repetition]:
                    repetitions[repetition][fold] = ([], [])

                type_ = line[0].decode('utf-8')
                if type_ == 'TRAIN':
                    repetitions[repetition][fold][0].append(line[1])
                elif type_ == 'TEST':
                    repetitions[repetition][fold][1].append(line[1])
                else:
                    raise ValueError(type_)

            for repetition in repetitions:
                for fold in repetitions[repetition]:
                    repetitions[repetition][fold] = Split(
                        np.array(repetitions[repetition][fold][0], dtype=np.int32),
                        np.array(repetitions[repetition][fold][1], dtype=np.int32))

            if cache:
                with open(pkl_filename, "wb") as fh:
                    pickle.dump({"name": name, "repetitions": repetitions}, fh,
                                protocol=2)

        return cls(name, '', repetitions)

    def from_dataset(self, X, Y, folds, repeats):
        raise NotImplementedError()

    def get(self, repeat=0, fold=0):
        if repeat not in self.split:
            raise ValueError("Repeat %s not known" % str(repeat))
        if fold not in self.split[repeat]:
            raise ValueError("Fold %s not known" % str(fold))
        return self.split[repeat][fold]

    def iterate_splits(self):
        for rep in range(self.repeats):
            yield (self.get(repeat=rep, fold=fold) for fold in range(self.folds))
