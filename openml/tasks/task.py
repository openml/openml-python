import os

from .. import datasets
from ..util import URLError
from .split import OpenMLSplit


class OpenMLTask(object):
    def __init__(self, task_id, task_type, data_set_id, target_feature,
                 estimation_procedure_type, data_splits_url,
                 estimation_parameters, evaluation_measure, cost_matrix,
                 api_connector, class_labels=None):
        self.task_id = int(task_id)
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.target_feature = target_feature
        self.estimation_procedure = dict()
        self.estimation_procedure["type"] = estimation_procedure_type
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.estimation_procedure["parameters"] = estimation_parameters
        #
        self.estimation_parameters = estimation_parameters
        self.evaluation_measure = evaluation_measure
        self.cost_matrix = cost_matrix
        self.api_connector = api_connector
        self.class_labels = class_labels

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")

    def __str__(self):
        return "OpenMLTask instance.\nTask ID: %s\n" \
               "Task type: %s\nDataset id: %s" \
               % (self.task_id, self.task_type, self.dataset_id)

    def get_dataset(self):
        """Download dataset associated with task"""
        return datasets.download_dataset(self.api_connector, self.dataset_id)

    def get_X_and_Y(self):
        dataset = self.get_dataset()
        # Replace with retrieve from cache
        if 'Supervised Classification'.lower() in self.task_type.lower():
            target_dtype = int
        elif 'Supervised Regression'.lower() in self.task_type.lower():
            target_dtype = float
        else:
            raise NotImplementedError(self.task_type)
        X_and_Y = dataset.get_dataset(target=self.target_feature,
                                      target_dtype=target_dtype)
        return X_and_Y

    def evaluate(self, algo):
        """Evaluate an algorithm on the test data.
        """
        raise NotImplementedError()

    def validate(self, algo):
        """Evaluate an algorithm on the validation data.
        """
        raise NotImplementedError()

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
            with open(cache_file):
                pass
        except (OSError, IOError):
            split_url = self.estimation_procedure["data_splits_url"]
            try:
                return_code, split_arff = self.api_connector._read_url(split_url)
            except (URLError, UnicodeEncodeError) as e:
                print(e, split_url)
                raise e

            with open(cache_file, "w") as fh:
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
            _create_task_cache_dir(self.api_connector, self.task_id), "datasplits.arff")

        try:
            split = OpenMLSplit.from_arff_file(cached_split_file)
        # Add FileNotFoundError in python3 version (which should be a
        # subclass of OSError.
        except (OSError, IOError):
            # Next, download and cache the associated split file
            self._download_split(cached_split_file)
            split = OpenMLSplit.from_arff_file(cached_split_file)

        return split


def _create_task_cache_dir(api_connector, task_id):
    task_cache_dir = os.path.join(api_connector.task_cache_dir, str(task_id))

    try:
        os.makedirs(task_cache_dir)
    except (IOError, OSError):
        # TODO add debug information!
        pass
    return task_cache_dir
