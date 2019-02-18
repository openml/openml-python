import io
import os
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse
from collections import OrderedDict

import xmltodict

from .. import datasets
from .split import OpenMLSplit
import openml._api_calls
from ..utils import _create_cache_directory_for_id, _tag_entity


class OpenMLTask(object):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 evaluation_measure):
        self.task_id = int(task_id)
        self.task_type_id = int(task_type_id)
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.evaluation_measure = evaluation_measure

    def get_dataset(self):
        """Download dataset associated with task"""
        return datasets.get_dataset(self.dataset_id)

    def get_train_test_split_indices(self, fold=0, repeat=0, sample=0):
        # Replace with retrieve from cache
        if self.split is None:
            self.split = self.download_split()

        train_indices, test_indices = self.split.get(
            repeat=repeat,
            fold=fold,
            sample=sample,
        )
        return train_indices, test_indices

    def _download_split(self, cache_file):
        try:
            with io.open(cache_file, encoding='utf8'):
                pass
        except (OSError, IOError):
            split_url = self.estimation_procedure["data_splits_url"]
            split_arff = openml._api_calls._read_url(split_url,
                                                     request_method='get')

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

        try:
            split = OpenMLSplit._from_arff_file(cached_split_file)
        except (OSError, IOError):
            # Next, download and cache the associated split file
            self._download_split(cached_split_file)
            split = OpenMLSplit._from_arff_file(cached_split_file)

        return split

    def get_split_dimensions(self):
        if self.split is None:
            self.split = self.download_split()

        return self.split.repeats, self.split.folds, self.split.samples

    def push_tag(self, tag):
        """Annotates this task with a tag on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the task.
        """
        _tag_entity('task', self.task_id, tag)

    def remove_tag(self, tag):
        """Removes a tag from this task on the server.

        Parameters
        ----------
        tag : str
            Tag to attach to the task.
        """
        _tag_entity('task', self.task_id, tag, untag=True)

    def _to_dict(self):

        task_container = OrderedDict()
        task_dict = OrderedDict([
            ('@xmlns:oml', 'http://openml.org/openml')
        ])

        task_container['oml:task_inputs'] = task_dict
        task_dict['oml:task_type_id'] = self.task_type_id

        source_data = OrderedDict([
            ('@name', 'source_data'),
            ('#text', str(self.dataset_id))
        ])

        return task_container, source_data

    def _to_xml(self):
        """Generate xml representation of self for upload to server.

        Returns
        -------
        str
            Task represented as XML string.
        """
        task_dict = self._to_dict()
        task_xml = xmltodict.unparse(task_dict, pretty=True)

        # A task may not be uploaded with the xml encoding specification:
        # <?xml version="1.0" encoding="utf-8"?>
        task_xml = task_xml.split('\n', 1)[-1]

        return task_xml

    def publish(self):
        """Publish task to OpenML server.

        Returns
        -------
        task_id: int
            Returns the id of the uploaded task
            if successful.

        """

        xml_description = self._to_xml()

        file_elements = {'description': xml_description}

        return_value = openml._api_calls._perform_api_call(
            "task/",
            file_elements=file_elements,
        )

        task_id = int(xmltodict.parse(return_value)['oml:upload_task']['oml:id'])

        return task_id


class OpenMLSupervisedTask(OpenMLTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 estimation_procedure_type, estimation_parameters,
                 evaluation_measure, target_name, data_splits_url,
                 estimation_procedure_id=1):
        super(OpenMLSupervisedTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            evaluation_measure=evaluation_measure,
        )
        self.estimation_procedure = dict()
        self.estimation_procedure["type"] = estimation_procedure_type
        self.estimation_procedure["parameters"] = estimation_parameters
        self.estimation_parameters = estimation_parameters
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.estimation_procedure_id = estimation_procedure_id
        self.target_name = target_name
        self.split = None

    def get_X_and_y(
        self,
        dataset_format: str = 'array',
    ) -> Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix]:
        """Get data associated with the current task.

        Parameters
        ----------
        dataset_format : str
            Data structure of the returned data. See :meth:`openml.datasets.OpenMLDataset.get_data`
            for possible options.

        Returns
        -------
        tuple - X and y

        """
        dataset = self.get_dataset()
        if self.task_type_id not in (1, 2, 3):
            raise NotImplementedError(self.task_type)
        X_and_y = dataset.get_data(
            dataset_format=dataset_format, target=self.target_name,
        )
        return X_and_y

    def _to_dict(self):

        task_container, source_data = super(OpenMLSupervisedTask, self)._to_dict()
        task_dict = task_container['oml:task_inputs']
        task_dict['oml:input'] = [source_data]

        # has an evaluation measure
        if self.evaluation_measure is not None:
            task_dict['oml:input'].append(
                OrderedDict([
                    ('@name', 'evaluation_measures'),
                    ('#text', self.evaluation_measure)
                ])
            )

        task_dict['oml:input'].extend(
            [
                OrderedDict([
                    ('@name', 'target_feature'),
                    ('#text', self.target_name)
                ]),

                OrderedDict([
                    ('@name', 'estimation_procedure'),
                    ('#text', str(self.estimation_procedure_id))
                ])
            ]
        )

        return task_container


class OpenMLClassificationTask(OpenMLSupervisedTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 estimation_procedure_type, estimation_parameters,
                 evaluation_measure, target_name, data_splits_url,
                 class_labels=None, cost_matrix=None,
                 estimation_procedure_id=1):
        super(OpenMLClassificationTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            evaluation_measure=evaluation_measure,
            target_name=target_name,
            data_splits_url=data_splits_url,
            estimation_procedure_id=estimation_procedure_id
        )
        self.class_labels = class_labels
        self.cost_matrix = cost_matrix

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")


class OpenMLRegressionTask(OpenMLSupervisedTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 estimation_procedure_type, estimation_parameters,
                 evaluation_measure, target_name, data_splits_url,
                 estimation_procedure_id=1):
        super(OpenMLRegressionTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            evaluation_measure=evaluation_measure,
            target_name=target_name,
            data_splits_url=data_splits_url,
            estimation_procedure_id=estimation_procedure_id
        )


class OpenMLClusteringTask(OpenMLTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 evaluation_measure):
        super(OpenMLClusteringTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            evaluation_measure=evaluation_measure,
        )

    def get_X(
        self,
        dataset_format: str = 'array',
    ) -> Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix]:
        """Get data associated with the current task.

        Parameters
        ----------
        dataset_format : str
            Data structure of the returned data. See :meth:`openml.datasets.OpenMLDataset.get_data`
            for possible options.

        Returns
        -------
        tuple - X and y

        """
        dataset = self.get_dataset()
        X_and_y = dataset.get_data(
            dataset_format=dataset_format, target=None,
        )
        return X_and_y

    def _to_dict(self):

        task_container, source_data = super(OpenMLClusteringTask, self)._to_dict()
        task_dict = task_container['oml:task_inputs']
        task_dict['oml:input'] = source_data

        return task_container


class OpenMLLearningCurveTask(OpenMLClassificationTask):
    def __init__(self, task_id, task_type_id, task_type, data_set_id,
                 estimation_procedure_type, estimation_parameters,
                 evaluation_measure, target_name, data_splits_url,
                 class_labels=None, cost_matrix=None,
                 estimation_procedure_id=1):
        super(OpenMLLearningCurveTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            evaluation_measure=evaluation_measure,
            target_name=target_name,
            data_splits_url=data_splits_url,
            class_labels=class_labels,
            cost_matrix=cost_matrix,
            estimation_procedure_id=estimation_procedure_id
        )
        self.target_name = target_name
        self.class_labels = class_labels
        self.cost_matrix = cost_matrix
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.split = None

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")


class TaskTypeEnum(object):
    SUPERVISED_CLASSIFICATION = 1
    SUPERVISED_REGRESSION = 2
    LEARNING_CURVE = 3
    SUPERVISED_DATASTREAM_CLASSIFICATION = 4
    CLUSTERING = 5
    MACHINE_LEARNING_CHALLENGE = 6
    SURVIVAL_ANALYSIS = 7
    SUBGROUP_DISCOVERY = 8
