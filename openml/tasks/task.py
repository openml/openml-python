import io
import os

from .. import config
from .. import datasets
from .split import OpenMLSplit
import openml._api_calls
from ..utils import _create_cache_directory_for_id

class OpenMLTask(object):
    def __init__(self, task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure):
        self.task_id = int(task_id)
        self.task_type_id = int(task_type_id)
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.estimation_procedure = dict()
        self.estimation_procedure["type"] = estimation_procedure_type
        self.estimation_procedure["parameters"] = estimation_parameters
        #
        self.estimation_parameters = estimation_parameters
        self.evaluation_measure = evaluation_measure

    def get_dataset(self):
        """Download dataset associated with task"""
        return datasets.get_dataset(self.dataset_id)

    def push_tag(self, tag):
        """Annotates this task with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the task.
        """
        data = {'task_id': self.task_id, 'tag': tag}
        openml._api_calls._perform_api_call("/task/tag", data=data)

    def remove_tag(self, tag):
        """Removes a tag from this task on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the task.
        """
        data = {'task_id': self.task_id, 'tag': tag}
        openml._api_calls._perform_api_call("/task/untag", data=data)

class SupervisedTask(OpenMLTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure, target_name, data_splits_url):
        super(SupervisedTask, self).__init__(task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure)
        self.target_name = target_name
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.split = None

    def get_X_and_y(self):
        """Get data associated with the current task.

        Returns
        -------
        tuple - X and y

        """
        dataset = self.get_dataset()
        if self.task_type_id not in (1, 2, 3):
            raise NotImplementedError(self.task_type)
        X_and_y = dataset.get_data(target=self.target_name)
        return X_and_y

    def get_train_test_split_indices(self, fold=0, repeat=0, sample=0):
        # Replace with retrieve from cache
        if self.split is None:
            self.split = self.download_split()

        train_indices, test_indices = self.split.get(repeat=repeat, fold=fold, sample=sample)
        return train_indices, test_indices

    def _download_split(self, cache_file):
        try:
            with io.open(cache_file, encoding='utf8'):
                pass
        except (OSError, IOError):
            split_url = self.estimation_procedure["data_splits_url"]
            split_arff = openml._api_calls._read_url(split_url)

            with io.open(cache_file, "w", encoding='utf8') as fh:
                fh.write(split_arff)
            del split_arff

    def download_split(self):
        """Download the OpenML split for a given task.
        """
        cached_split_file = os.path.join(
            _create_cache_directory_for_id('tasks', self.task_id),
            "datasplits.arff",
        )

        # Not all tasks come with a split, e.g. in clustering the full dataset is always used
        if self.estimation_procedure["data_splits_url"]:

            try:
                split = OpenMLSplit._from_arff_file(cached_split_file)
            except (OSError, IOError):
                # Next, download and cache the associated split file
                self._download_split(cached_split_file)
                split = OpenMLSplit._from_arff_file(cached_split_file)

            return split

        else:  # if no data splits are used
            no_split = {0: {0: {0: (list(range(self.get_dataset().get_data().shape[0])),
                                    list(range(self.get_dataset().get_data().shape[0])))}}}
            split = OpenMLSplit('no_split', 'no actual split, all points in train and test', no_split)
            return split

    def get_split_dimensions(self):
        if self.split is None:
            self.split = self.download_split()

        return self.split.repeats, self.split.folds, self.split.samples



class ClassificationTask(SupervisedTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure, target_name, data_splits_url, class_labels=None, cost_matrix=None):
        super(ClassificationTask, self).__init__(task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure, target_name, data_splits_url)
        self.target_name = target_name
        self.class_labels = class_labels
        self.cost_matrix = cost_matrix
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.split = None

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")

class RegressionTask(SupervisedTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure, target_name, data_splits_url):
        super(RegressionTask, self).__init__(task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure, target_name, data_splits_url)


class ClusteringTask(OpenMLTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure, number_of_clusters=None):
        super(ClusteringTask, self).__init__(task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure)
        self.number_of_clusters = number_of_clusters

class LearningCurveTask(SupervisedTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                 estimation_parameters, evaluation_measure, target_name, data_splits_url, class_labels=None,
                 cost_matrix=None):
        super(LearningCurveTask, self).__init__(task_id, task_type_id, task_type, data_set_id, estimation_procedure_type,
                         estimation_parameters, evaluation_measure, target_name, data_splits_url)
        self.target_name = target_name
        self.class_labels = class_labels
        self.cost_matrix = cost_matrix
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.split = None

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")







