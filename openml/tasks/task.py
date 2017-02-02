import io
import os

from .. import config
from .. import datasets
from ..util import URLError
from .split import OpenMLSplit
from .._api_calls import _read_url


class OpenMLTask(object):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 target_name, estimation_procedure_type, data_splits_url,
                 estimation_parameters, evaluation_measure, cost_matrix,
                 class_labels=None):
        self.task_id = int(task_id)
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.target_name = target_name
        self.estimation_procedure = dict()
        self.estimation_procedure["type"] = estimation_procedure_type
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.estimation_procedure["parameters"] = estimation_parameters
        #
        self.estimation_parameters = estimation_parameters
        self.evaluation_measure = evaluation_measure
        self.cost_matrix = cost_matrix
        self.class_labels = class_labels

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")

    def get_dataset(self):
        """Download dataset associated with task"""
        return datasets.get_dataset(self.dataset_id)

    def get_X_and_y(self):
        dataset = self.get_dataset()
        # Replace with retrieve from cache
        if 'Supervised Classification'.lower() in self.task_type.lower():
            target_dtype = int
        elif 'Supervised Regression'.lower() in self.task_type.lower():
            target_dtype = float
        else:
            raise NotImplementedError(self.task_type)
        X_and_y = dataset.get_data(target=self.target_name,
                                   target_dtype=target_dtype)
        return X_and_y

    def get_train_test_split_indices(self, fold=0, repeat=0):
        # Replace with retrieve from cache
        split = self.download_split()
        train_indices, test_indices = split.get(repeat=repeat, fold=fold)
        return train_indices, test_indices

    def iterate_repeats(self):
        split = self.download_split()
        for rep in split.iterate_splits():
            yield rep

    def iterate_all_splits(self):
        split = self.download_split()
        for rep in split.iterate_splits():
            for fold in rep:
                yield fold

    def _download_split(self, cache_file):
        try:
            with io.open(cache_file, encoding='utf8'):
                pass
        except (OSError, IOError):
            split_url = self.estimation_procedure["data_splits_url"]
            try:
                split_arff = _read_url(split_url)
            except (URLError, UnicodeEncodeError) as e:
                print(e, split_url)
                raise e

            with io.open(cache_file, "w", encoding='utf8') as fh:
                fh.write(split_arff)
            del split_arff

    def download_split(self):
        """Download the OpenML split for a given task.

        Parameters
        ----------
        task_id : Task
            An entity of :class:`openml.OpenMLTask`.
        """
        cached_split_file = os.path.join(
            _create_task_cache_dir(self.task_id), "datasplits.arff")

        try:
            split = OpenMLSplit._from_arff_file(cached_split_file)
        # Add FileNotFoundError in python3 version (which should be a
        # subclass of OSError.
        except (OSError, IOError):
            # Next, download and cache the associated split file
            self._download_split(cached_split_file)
            split = OpenMLSplit._from_arff_file(cached_split_file)

        return split


def _create_task_cache_dir(task_id):
    task_cache_dir = os.path.join(config.get_cache_directory(), "tasks", str(task_id))

    try:
        os.makedirs(task_cache_dir)
    except (IOError, OSError):
        # TODO add debug information!
        pass
    return task_cache_dir
