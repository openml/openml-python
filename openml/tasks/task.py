# License: BSD 3-Clause

from abc import ABC
from collections import OrderedDict
from enum import Enum
import io
import os
from typing import Union, Tuple, Dict, List, Optional, Any
from warnings import warn

import numpy as np
import pandas as pd
import scipy.sparse

import openml._api_calls
from openml.base import OpenMLBase
from .. import datasets
from .split import OpenMLSplit
from ..utils import _create_cache_directory_for_id


class TaskType(Enum):
    SUPERVISED_CLASSIFICATION = 1
    SUPERVISED_REGRESSION = 2
    LEARNING_CURVE = 3
    SUPERVISED_DATASTREAM_CLASSIFICATION = 4
    CLUSTERING = 5
    MACHINE_LEARNING_CHALLENGE = 6
    SURVIVAL_ANALYSIS = 7
    SUBGROUP_DISCOVERY = 8
    MULTITASK_REGRESSION = 9


class OpenMLTask(OpenMLBase):
    """OpenML Task object.

       Parameters
       ----------
       task_type_id : TaskType
           Refers to the type of task.
       task_type : str
           Refers to the task.
       data_set_id: int
           Refers to the data.
       estimation_procedure_id: int
           Refers to the type of estimates used.
    """

    def __init__(
        self,
        task_id: Optional[int],
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        estimation_procedure_id: int = 1,
        estimation_procedure_type: Optional[str] = None,
        estimation_parameters: Optional[Dict[str, str]] = None,
        evaluation_measure: Optional[str] = None,
        data_splits_url: Optional[str] = None,
    ):

        self.task_id = int(task_id) if task_id is not None else None
        self.task_type_id = task_type_id
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.evaluation_measure = evaluation_measure
        self.estimation_procedure = (
            dict()
        )  # type: Dict[str, Optional[Union[str, Dict]]] # noqa E501
        self.estimation_procedure["type"] = estimation_procedure_type
        self.estimation_procedure["parameters"] = estimation_parameters
        self.estimation_procedure["data_splits_url"] = data_splits_url
        self.estimation_procedure_id = estimation_procedure_id
        self.split = None  # type: Optional[OpenMLSplit]

    @classmethod
    def _entity_letter(cls) -> str:
        return "t"

    @property
    def id(self) -> Optional[int]:
        return self.task_id

    def _get_repr_body_fields(self) -> List[Tuple[str, Union[str, int, List[str]]]]:
        """ Collect all information to display in the __repr__ body. """
        fields = {
            "Task Type Description": "{}/tt/{}".format(
                openml.config.get_server_base_url(), self.task_type_id
            )
        }  # type: Dict[str, Any]
        if self.task_id is not None:
            fields["Task ID"] = self.task_id
            fields["Task URL"] = self.openml_url
        if self.evaluation_measure is not None:
            fields["Evaluation Measure"] = self.evaluation_measure
        if self.estimation_procedure is not None:
            fields["Estimation Procedure"] = self.estimation_procedure["type"]
        if getattr(self, "target_name", None) is not None:
            fields["Target Feature"] = getattr(self, "target_name")
            if hasattr(self, "class_labels"):
                fields["# of Classes"] = len(getattr(self, "class_labels"))
            if hasattr(self, "cost_matrix"):
                fields["Cost Matrix"] = "Available"

        # determines the order in which the information will be printed
        order = [
            "Task Type Description",
            "Task ID",
            "Task URL",
            "Estimation Procedure",
            "Evaluation Measure",
            "Target Feature",
            "# of Classes",
            "Cost Matrix",
        ]
        return [(key, fields[key]) for key in order if key in fields]

    def get_dataset(self) -> datasets.OpenMLDataset:
        """Download dataset associated with task"""
        return datasets.get_dataset(self.dataset_id)

    def get_train_test_split_indices(
        self, fold: int = 0, repeat: int = 0, sample: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Replace with retrieve from cache
        if self.split is None:
            self.split = self.download_split()

        train_indices, test_indices = self.split.get(repeat=repeat, fold=fold, sample=sample,)
        return train_indices, test_indices

    def _download_split(self, cache_file: str):
        try:
            with io.open(cache_file, encoding="utf8"):
                pass
        except (OSError, IOError):
            split_url = self.estimation_procedure["data_splits_url"]
            openml._api_calls._download_text_file(
                source=str(split_url), output_path=cache_file,
            )

    def download_split(self) -> OpenMLSplit:
        """Download the OpenML split for a given task.
        """
        cached_split_file = os.path.join(
            _create_cache_directory_for_id("tasks", self.task_id), "datasplits.arff",
        )

        try:
            split = OpenMLSplit._from_arff_file(cached_split_file)
        except (OSError, IOError):
            # Next, download and cache the associated split file
            self._download_split(cached_split_file)
            split = OpenMLSplit._from_arff_file(cached_split_file)

        return split

    def get_split_dimensions(self) -> Tuple[int, int, int]:

        if self.split is None:
            self.split = self.download_split()

        return self.split.repeats, self.split.folds, self.split.samples

    def _to_dict(self) -> "OrderedDict[str, OrderedDict]":
        """ Creates a dictionary representation of self. """
        task_container = OrderedDict()  # type: OrderedDict[str, OrderedDict]
        task_dict = OrderedDict(
            [("@xmlns:oml", "http://openml.org/openml")]
        )  # type: OrderedDict[str, Union[List, str, TaskType]]

        task_container["oml:task_inputs"] = task_dict
        task_dict["oml:task_type_id"] = self.task_type_id.value

        # having task_inputs and adding a type annotation
        # solves wrong warnings
        task_inputs = [
            OrderedDict([("@name", "source_data"), ("#text", str(self.dataset_id))]),
            OrderedDict(
                [("@name", "estimation_procedure"), ("#text", str(self.estimation_procedure_id))]
            ),
        ]  # type: List[OrderedDict]

        if self.evaluation_measure is not None:
            task_inputs.append(
                OrderedDict([("@name", "evaluation_measures"), ("#text", self.evaluation_measure)])
            )

        task_dict["oml:input"] = task_inputs

        return task_container

    def _parse_publish_response(self, xml_response: Dict):
        """ Parse the id from the xml_response and assign it to self. """
        self.task_id = int(xml_response["oml:upload_task"]["oml:id"])


class OpenMLSupervisedTask(OpenMLTask, ABC):
    """OpenML Supervised Classification object.

       Inherited from :class:`openml.OpenMLTask`

       Parameters
       ----------
       target_name : str
           Name of the target feature (the class variable).
    """

    def __init__(
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 1,
        estimation_procedure_type: Optional[str] = None,
        estimation_parameters: Optional[Dict[str, str]] = None,
        evaluation_measure: Optional[str] = None,
        data_splits_url: Optional[str] = None,
        task_id: Optional[int] = None,
    ):
        super(OpenMLSupervisedTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            estimation_procedure_id=estimation_procedure_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            evaluation_measure=evaluation_measure,
            data_splits_url=data_splits_url,
        )

        self.target_name = target_name

    def get_X_and_y(
        self, dataset_format: str = "array",
    ) -> Tuple[
        Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix], Union[np.ndarray, pd.Series]
    ]:
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
        if self.task_type_id not in (
            TaskType.SUPERVISED_CLASSIFICATION,
            TaskType.SUPERVISED_REGRESSION,
            TaskType.LEARNING_CURVE,
        ):
            raise NotImplementedError(self.task_type)
        X, y, _, _ = dataset.get_data(dataset_format=dataset_format, target=self.target_name,)
        return X, y

    def _to_dict(self) -> "OrderedDict[str, OrderedDict]":

        task_container = super(OpenMLSupervisedTask, self)._to_dict()
        task_dict = task_container["oml:task_inputs"]

        task_dict["oml:input"].append(
            OrderedDict([("@name", "target_feature"), ("#text", self.target_name)])
        )

        return task_container

    @property
    def estimation_parameters(self):

        warn(
            "The estimation_parameters attribute will be "
            "deprecated in the future, please use "
            "estimation_procedure['parameters'] instead",
            PendingDeprecationWarning,
        )
        return self.estimation_procedure["parameters"]

    @estimation_parameters.setter
    def estimation_parameters(self, est_parameters):

        self.estimation_procedure["parameters"] = est_parameters


class OpenMLClassificationTask(OpenMLSupervisedTask):
    """OpenML Classification object.

       Inherited from :class:`openml.OpenMLSupervisedTask`

       Parameters
       ----------
       class_labels : List of str (optional)
       cost_matrix: array (optional)
    """

    def __init__(
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 1,
        estimation_procedure_type: Optional[str] = None,
        estimation_parameters: Optional[Dict[str, str]] = None,
        evaluation_measure: Optional[str] = None,
        data_splits_url: Optional[str] = None,
        task_id: Optional[int] = None,
        class_labels: Optional[List[str]] = None,
        cost_matrix: Optional[np.ndarray] = None,
    ):

        super(OpenMLClassificationTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            estimation_procedure_id=estimation_procedure_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            evaluation_measure=evaluation_measure,
            target_name=target_name,
            data_splits_url=data_splits_url,
        )
        self.class_labels = class_labels
        self.cost_matrix = cost_matrix

        if cost_matrix is not None:
            raise NotImplementedError("Costmatrix")


class OpenMLRegressionTask(OpenMLSupervisedTask):
    """OpenML Regression object.

       Inherited from :class:`openml.OpenMLSupervisedTask`
    """

    def __init__(
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 7,
        estimation_procedure_type: Optional[str] = None,
        estimation_parameters: Optional[Dict[str, str]] = None,
        data_splits_url: Optional[str] = None,
        task_id: Optional[int] = None,
        evaluation_measure: Optional[str] = None,
    ):
        super(OpenMLRegressionTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            estimation_procedure_id=estimation_procedure_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            evaluation_measure=evaluation_measure,
            target_name=target_name,
            data_splits_url=data_splits_url,
        )


class OpenMLClusteringTask(OpenMLTask):
    """OpenML Clustering object.

       Inherited from :class:`openml.OpenMLTask`

       Parameters
       ----------
       target_name : str (optional)
           Name of the target feature (class) that is not part of the
           feature set for the clustering task.
    """

    def __init__(
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        estimation_procedure_id: int = 17,
        task_id: Optional[int] = None,
        estimation_procedure_type: Optional[str] = None,
        estimation_parameters: Optional[Dict[str, str]] = None,
        data_splits_url: Optional[str] = None,
        evaluation_measure: Optional[str] = None,
        target_name: Optional[str] = None,
    ):
        super(OpenMLClusteringTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            evaluation_measure=evaluation_measure,
            estimation_procedure_id=estimation_procedure_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            data_splits_url=data_splits_url,
        )

        self.target_name = target_name

    def get_X(
        self, dataset_format: str = "array",
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
        data, *_ = dataset.get_data(dataset_format=dataset_format, target=None,)
        return data

    def _to_dict(self) -> "OrderedDict[str, OrderedDict]":

        task_container = super(OpenMLClusteringTask, self)._to_dict()

        # Right now, it is not supported as a feature.
        # Uncomment if it is supported on the server
        # in the future.
        # https://github.com/openml/OpenML/issues/925
        """
        task_dict = task_container['oml:task_inputs']
        if self.target_name is not None:
            task_dict['oml:input'].append(
                OrderedDict([
                    ('@name', 'target_feature'),
                    ('#text', self.target_name)
                ])
            )
        """
        return task_container


class OpenMLLearningCurveTask(OpenMLClassificationTask):
    """OpenML Learning Curve object.

       Inherited from :class:`openml.OpenMLClassificationTask`
    """

    def __init__(
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 13,
        estimation_procedure_type: Optional[str] = None,
        estimation_parameters: Optional[Dict[str, str]] = None,
        data_splits_url: Optional[str] = None,
        task_id: Optional[int] = None,
        evaluation_measure: Optional[str] = None,
        class_labels: Optional[List[str]] = None,
        cost_matrix: Optional[np.ndarray] = None,
    ):
        super(OpenMLLearningCurveTask, self).__init__(
            task_id=task_id,
            task_type_id=task_type_id,
            task_type=task_type,
            data_set_id=data_set_id,
            estimation_procedure_id=estimation_procedure_id,
            estimation_procedure_type=estimation_procedure_type,
            estimation_parameters=estimation_parameters,
            evaluation_measure=evaluation_measure,
            target_name=target_name,
            data_splits_url=data_splits_url,
            class_labels=class_labels,
            cost_matrix=cost_matrix,
        )
