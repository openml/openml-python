# License: BSD 3-Clause
# TODO(eddbergman): Seems like a lot of the subclasses could just get away with setting
# a `ClassVar` for whatever changes as their `__init__` defaults, less duplicated code.
from __future__ import annotations

import warnings
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from typing_extensions import Literal, TypedDict, overload

import openml._api_calls
import openml.config
from openml import datasets
from openml.base import OpenMLBase
from openml.utils import _create_cache_directory_for_id

from .split import OpenMLSplit

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import scipy.sparse


# TODO(eddiebergman): Should use `auto()` but might be too late if these numbers are used
# and stored on server.
class TaskType(Enum):
    """Possible task types as defined in OpenML."""

    SUPERVISED_CLASSIFICATION = 1
    SUPERVISED_REGRESSION = 2
    LEARNING_CURVE = 3
    SUPERVISED_DATASTREAM_CLASSIFICATION = 4
    CLUSTERING = 5
    MACHINE_LEARNING_CHALLENGE = 6
    SURVIVAL_ANALYSIS = 7
    SUBGROUP_DISCOVERY = 8
    MULTITASK_REGRESSION = 9


class _EstimationProcedure(TypedDict):
    type: str | None
    parameters: dict[str, str] | None
    data_splits_url: str | None


class OpenMLTask(OpenMLBase):
    """OpenML Task object.

    Parameters
    ----------
    task_id: Union[int, None]
        Refers to the unique identifier of OpenML task.
    task_type_id: TaskType
        Refers to the type of OpenML task.
    task_type: str
        Refers to the OpenML task.
    data_set_id: int
        Refers to the data.
    estimation_procedure_id: int
        Refers to the type of estimates used.
    estimation_procedure_type: str, default=None
        Refers to the type of estimation procedure used for the OpenML task.
    estimation_parameters: [Dict[str, str]], default=None
        Estimation parameters used for the OpenML task.
    evaluation_measure: str, default=None
        Refers to the evaluation measure.
    data_splits_url: str, default=None
        Refers to the URL of the data splits used for the OpenML task.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_id: int | None,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        estimation_procedure_id: int = 1,
        estimation_procedure_type: str | None = None,
        estimation_parameters: dict[str, str] | None = None,
        evaluation_measure: str | None = None,
        data_splits_url: str | None = None,
    ):
        self.task_id = int(task_id) if task_id is not None else None
        self.task_type_id = task_type_id
        self.task_type = task_type
        self.dataset_id = int(data_set_id)
        self.evaluation_measure = evaluation_measure
        self.estimation_procedure: _EstimationProcedure = {
            "type": estimation_procedure_type,
            "parameters": estimation_parameters,
            "data_splits_url": data_splits_url,
        }
        self.estimation_procedure_id = estimation_procedure_id
        self.split: OpenMLSplit | None = None

    @classmethod
    def _entity_letter(cls) -> str:
        return "t"

    @property
    def id(self) -> int | None:
        """Return the OpenML ID of this task."""
        return self.task_id

    def _get_repr_body_fields(self) -> Sequence[tuple[str, str | int | list[str]]]:
        """Collect all information to display in the __repr__ body."""
        base_server_url = openml.config.get_server_base_url()
        fields: dict[str, Any] = {
            "Task Type Description": f"{base_server_url}/tt/{self.task_type_id}"
        }
        if self.task_id is not None:
            fields["Task ID"] = self.task_id
            fields["Task URL"] = self.openml_url
        if self.evaluation_measure is not None:
            fields["Evaluation Measure"] = self.evaluation_measure
        if self.estimation_procedure is not None:
            fields["Estimation Procedure"] = self.estimation_procedure["type"]

        # TODO(eddiebergman): Subclasses could advertise/provide this, instead of having to
        # have the base class know about it's subclasses.
        target_name = getattr(self, "target_name", None)
        if target_name is not None:
            fields["Target Feature"] = target_name

            class_labels = getattr(self, "class_labels", None)
            if class_labels is not None:
                fields["# of Classes"] = len(class_labels)

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

    def get_dataset(self, **kwargs: Any) -> datasets.OpenMLDataset:
        """Download dataset associated with task.

        Accepts the same keyword arguments as the `openml.datasets.get_dataset`.
        """
        return datasets.get_dataset(self.dataset_id, **kwargs)

    def get_train_test_split_indices(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the indices of the train and test splits for a given task."""
        # Replace with retrieve from cache
        if self.split is None:
            self.split = self.download_split()

        return self.split.get(repeat=repeat, fold=fold, sample=sample)

    def _download_split(self, cache_file: Path) -> None:
        # TODO(eddiebergman): Not sure about this try to read and error approach
        try:
            with cache_file.open(encoding="utf8"):
                pass
        except OSError:
            split_url = self.estimation_procedure["data_splits_url"]
            openml._api_calls._download_text_file(
                source=str(split_url),
                output_path=str(cache_file),
            )

    def download_split(self) -> OpenMLSplit:
        """Download the OpenML split for a given task."""
        # TODO(eddiebergman): Can this every be `None`?
        assert self.task_id is not None
        cache_dir = _create_cache_directory_for_id("tasks", self.task_id)
        cached_split_file = cache_dir / "datasplits.arff"

        try:
            split = OpenMLSplit._from_arff_file(cached_split_file)
        except OSError:
            # Next, download and cache the associated split file
            self._download_split(cached_split_file)
            split = OpenMLSplit._from_arff_file(cached_split_file)

        return split

    def get_split_dimensions(self) -> tuple[int, int, int]:
        """Get the (repeats, folds, samples) of the split for a given task."""
        if self.split is None:
            self.split = self.download_split()

        return self.split.repeats, self.split.folds, self.split.samples

    # TODO(eddiebergman): Really need some better typing on all this
    def _to_dict(self) -> dict[str, dict[str, int | str | list[dict[str, Any]]]]:
        """Creates a dictionary representation of self in a string format (for XML parsing)."""
        oml_input = [
            {"@name": "source_data", "#text": str(self.dataset_id)},
            {"@name": "estimation_procedure", "#text": str(self.estimation_procedure_id)},
        ]
        if self.evaluation_measure is not None:
            oml_input.append({"@name": "evaluation_measures", "#text": self.evaluation_measure})

        return {
            "oml:task_inputs": {
                "@xmlns:oml": "http://openml.org/openml",
                "oml:task_type_id": self.task_type_id.value,  # This is an int from the enum?
                "oml:input": oml_input,
            }
        }

    def _parse_publish_response(self, xml_response: dict) -> None:
        """Parse the id from the xml_response and assign it to self."""
        self.task_id = int(xml_response["oml:upload_task"]["oml:id"])


class OpenMLSupervisedTask(OpenMLTask, ABC):
    """OpenML Supervised Classification object.

    Parameters
    ----------
    task_type_id : TaskType
        ID of the task type.
    task_type : str
        Name of the task type.
    data_set_id : int
        ID of the OpenML dataset associated with the task.
    target_name : str
        Name of the target feature (the class variable).
    estimation_procedure_id : int, default=None
        ID of the estimation procedure for the task.
    estimation_procedure_type : str, default=None
        Type of the estimation procedure for the task.
    estimation_parameters : dict, default=None
        Estimation parameters for the task.
    evaluation_measure : str, default=None
        Name of the evaluation measure for the task.
    data_splits_url : str, default=None
        URL of the data splits for the task.
    task_id: Union[int, None]
        Refers to the unique identifier of task.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 1,
        estimation_procedure_type: str | None = None,
        estimation_parameters: dict[str, str] | None = None,
        evaluation_measure: str | None = None,
        data_splits_url: str | None = None,
        task_id: int | None = None,
    ):
        super().__init__(
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

    @overload
    def get_X_and_y(
        self, dataset_format: Literal["array"] = "array"
    ) -> tuple[
        np.ndarray | scipy.sparse.spmatrix,
        np.ndarray | None,
    ]: ...

    @overload
    def get_X_and_y(
        self, dataset_format: Literal["dataframe"]
    ) -> tuple[
        pd.DataFrame,
        pd.Series | pd.DataFrame | None,
    ]: ...

    # TODO(eddiebergman): Do all OpenMLSupervisedTask have a `y`?
    def get_X_and_y(
        self, dataset_format: Literal["dataframe", "array"] = "array"
    ) -> tuple[
        np.ndarray | pd.DataFrame | scipy.sparse.spmatrix,
        np.ndarray | pd.Series | pd.DataFrame | None,
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
        # TODO: [0.15]
        if dataset_format == "array":
            warnings.warn(
                "Support for `dataset_format='array'` will be removed in 0.15,"
                "start using `dataset_format='dataframe' to ensure your code "
                "will continue to work. You can use the dataframe's `to_numpy` "
                "function to continue using numpy arrays.",
                category=FutureWarning,
                stacklevel=2,
            )
        dataset = self.get_dataset()
        if self.task_type_id not in (
            TaskType.SUPERVISED_CLASSIFICATION,
            TaskType.SUPERVISED_REGRESSION,
            TaskType.LEARNING_CURVE,
        ):
            raise NotImplementedError(self.task_type)

        X, y, _, _ = dataset.get_data(
            dataset_format=dataset_format,
            target=self.target_name,
        )
        return X, y

    def _to_dict(self) -> dict[str, dict]:
        task_container = super()._to_dict()
        oml_input = task_container["oml:task_inputs"]["oml:input"]  # type: ignore
        assert isinstance(oml_input, list)

        oml_input.append({"@name": "target_feature", "#text": self.target_name})
        return task_container

    @property
    def estimation_parameters(self) -> dict[str, str] | None:
        """Return the estimation parameters for the task."""
        warnings.warn(
            "The estimation_parameters attribute will be "
            "deprecated in the future, please use "
            "estimation_procedure['parameters'] instead",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.estimation_procedure["parameters"]

    @estimation_parameters.setter
    def estimation_parameters(self, est_parameters: dict[str, str] | None) -> None:
        self.estimation_procedure["parameters"] = est_parameters


class OpenMLClassificationTask(OpenMLSupervisedTask):
    """OpenML Classification object.

    Parameters
    ----------
    task_type_id : TaskType
        ID of the Classification task type.
    task_type : str
        Name of the Classification task type.
    data_set_id : int
        ID of the OpenML dataset associated with the Classification task.
    target_name : str
        Name of the target variable.
    estimation_procedure_id : int, default=None
        ID of the estimation procedure for the Classification task.
    estimation_procedure_type : str, default=None
        Type of the estimation procedure.
    estimation_parameters : dict, default=None
        Estimation parameters for the Classification task.
    evaluation_measure : str, default=None
        Name of the evaluation measure.
    data_splits_url : str, default=None
        URL of the data splits for the Classification task.
    task_id : Union[int, None]
        ID of the Classification task (if it already exists on OpenML).
    class_labels : List of str, default=None
        A list of class labels (for classification tasks).
    cost_matrix : array, default=None
        A cost matrix (for classification tasks).
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 1,
        estimation_procedure_type: str | None = None,
        estimation_parameters: dict[str, str] | None = None,
        evaluation_measure: str | None = None,
        data_splits_url: str | None = None,
        task_id: int | None = None,
        class_labels: list[str] | None = None,
        cost_matrix: np.ndarray | None = None,
    ):
        super().__init__(
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

    Parameters
    ----------
    task_type_id : TaskType
        Task type ID of the OpenML Regression task.
    task_type : str
        Task type of the OpenML Regression task.
    data_set_id : int
        ID of the OpenML dataset.
    target_name : str
        Name of the target feature used in the Regression task.
    estimation_procedure_id : int, default=None
        ID of the OpenML estimation procedure.
    estimation_procedure_type : str, default=None
        Type of the OpenML estimation procedure.
    estimation_parameters : dict, default=None
        Parameters used by the OpenML estimation procedure.
    data_splits_url : str, default=None
        URL of the OpenML data splits for the Regression task.
    task_id : Union[int, None]
        ID of the OpenML Regression task.
    evaluation_measure : str, default=None
        Evaluation measure used in the Regression task.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 7,
        estimation_procedure_type: str | None = None,
        estimation_parameters: dict[str, str] | None = None,
        data_splits_url: str | None = None,
        task_id: int | None = None,
        evaluation_measure: str | None = None,
    ):
        super().__init__(
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

    Parameters
    ----------
    task_type_id : TaskType
        Task type ID of the OpenML clustering task.
    task_type : str
        Task type of the OpenML clustering task.
    data_set_id : int
        ID of the OpenML dataset used in clustering the task.
    estimation_procedure_id : int, default=None
        ID of the OpenML estimation procedure.
    task_id : Union[int, None]
        ID of the OpenML clustering task.
    estimation_procedure_type : str, default=None
        Type of the OpenML estimation procedure used in the clustering task.
    estimation_parameters : dict, default=None
        Parameters used by the OpenML estimation procedure.
    data_splits_url : str, default=None
        URL of the OpenML data splits for the clustering task.
    evaluation_measure : str, default=None
        Evaluation measure used in the clustering task.
    target_name : str, default=None
        Name of the target feature (class) that is not part of the
        feature set for the clustering task.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        estimation_procedure_id: int = 17,
        task_id: int | None = None,
        estimation_procedure_type: str | None = None,
        estimation_parameters: dict[str, str] | None = None,
        data_splits_url: str | None = None,
        evaluation_measure: str | None = None,
        target_name: str | None = None,
    ):
        super().__init__(
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

    @overload
    def get_X(
        self,
        dataset_format: Literal["array"] = "array",
    ) -> np.ndarray | scipy.sparse.spmatrix: ...

    @overload
    def get_X(self, dataset_format: Literal["dataframe"]) -> pd.DataFrame: ...

    def get_X(
        self,
        dataset_format: Literal["array", "dataframe"] = "array",
    ) -> np.ndarray | pd.DataFrame | scipy.sparse.spmatrix:
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
        data, *_ = dataset.get_data(dataset_format=dataset_format, target=None)
        return data

    def _to_dict(self) -> dict[str, dict[str, int | str | list[dict[str, Any]]]]:
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
        return super()._to_dict()


class OpenMLLearningCurveTask(OpenMLClassificationTask):
    """OpenML Learning Curve object.

    Parameters
    ----------
    task_type_id : TaskType
        ID of the Learning Curve task.
    task_type : str
        Name of the Learning Curve task.
    data_set_id : int
        ID of the dataset that this task is associated with.
    target_name : str
        Name of the target feature in the dataset.
    estimation_procedure_id : int, default=None
        ID of the estimation procedure to use for evaluating models.
    estimation_procedure_type : str, default=None
        Type of the estimation procedure.
    estimation_parameters : dict, default=None
        Additional parameters for the estimation procedure.
    data_splits_url : str, default=None
        URL of the file containing the data splits for Learning Curve task.
    task_id : Union[int, None]
        ID of the Learning Curve task.
    evaluation_measure : str, default=None
        Name of the evaluation measure to use for evaluating models.
    class_labels : list of str, default=None
        Class labels for Learning Curve tasks.
    cost_matrix : numpy array, default=None
        Cost matrix for Learning Curve tasks.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type_id: TaskType,
        task_type: str,
        data_set_id: int,
        target_name: str,
        estimation_procedure_id: int = 13,
        estimation_procedure_type: str | None = None,
        estimation_parameters: dict[str, str] | None = None,
        data_splits_url: str | None = None,
        task_id: int | None = None,
        evaluation_measure: str | None = None,
        class_labels: list[str] | None = None,
        cost_matrix: np.ndarray | None = None,
    ):
        super().__init__(
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
