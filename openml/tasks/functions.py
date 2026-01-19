# License: BSD 3-Clause
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd
from yaml import warnings

from openml._api.resources.tasks import TasksV1
from openml.datasets import get_dataset
import openml.utils
from openml._api import api_context

from .task import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    TaskType,
)

if TYPE_CHECKING:
    from .task import (
        OpenMLTask,
    )


def list_tasks(  # noqa: PLR0913
    task_type: TaskType | None = None,
    offset: int | None = None,
    size: int | None = None,
    tag: str | None = None,
    data_tag: str | None = None,
    status: str | None = None,
    data_name: str | None = None,
    data_id: int | None = None,
    number_instances: int | None = None,
    number_features: int | None = None,
    number_classes: int | None = None,
    number_missing_values: int | None = None,
) -> pd.DataFrame:
    """
    Return a number of tasks having the given tag and task_type

    Parameters
    ----------
    Filter task_type is separated from the other filters because
    it is used as task_type in the task description, but it is named
    type when used as a filter in list tasks call.
    offset : int, optional
        the number of tasks to skip, starting from the first
    task_type : TaskType, optional
        Refers to the type of task.
    size : int, optional
        the maximum number of tasks to show
    tag : str, optional
        the tag to include
    data_tag : str, optional
        the tag of the dataset
    data_id : int, optional
    status : str, optional
    data_name : str, optional
    number_instances : int, optional
    number_features : int, optional
    number_classes : int, optional
    number_missing_values : int, optional

    Returns
    -------
    dataframe
        All tasks having the given task_type and the give tag. Every task is
        represented by a row in the data frame containing the following information
        as columns: task id, dataset id, task_type and status. If qualities are
        calculated for the associated dataset, some of these are also returned.
    """
    listing_call = partial(
        api_context.backend.tasks.list,
        task_type=task_type,
        tag=tag,
        data_tag=data_tag,
        status=status,
        data_id=data_id,
        data_name=data_name,
        number_instances=number_instances,
        number_features=number_features,
        number_classes=number_classes,
        number_missing_values=number_missing_values,
    )
    batches = openml.utils._list_all(listing_call, offset=offset, limit=size)
    if len(batches) == 0:
        return pd.DataFrame()

    return pd.concat(batches)


def get_tasks(
    task_ids: list[int],
    download_data: bool | None = None,
    download_qualities: bool | None = None,
) -> list[OpenMLTask]:
    """Download tasks.

    This function iterates :meth:`openml.tasks.get`.

    Parameters
    ----------
    task_ids : List[int]
        A list of task ids to download.
    download_data : bool (default = True)
        Option to trigger download of data along with the meta data.
    download_qualities : bool (default=True)
        Option to download 'qualities' meta-data in addition to the minimal dataset description.

    Returns
    -------
    list
    """
    if download_data is None:
        warnings.warn(
            "`download_data` will default to False starting in 0.16. "
            "Please set `download_data` explicitly to suppress this warning.",
            stacklevel=1,
        )
        download_data = True

    if download_qualities is None:
        warnings.warn(
            "`download_qualities` will default to False starting in 0.16. "
            "Please set `download_qualities` explicitly to suppress this warning.",
            stacklevel=1,
        )
        download_qualities = True

    tasks = []
    for task_id in task_ids:
        tasks.append(
            api_context.backend.tasks.get(
                task_id, download_data=download_data, download_qualities=download_qualities
            )
        )
    return tasks


def get_task(
    task_id: int,
    download_splits: bool = False,  # noqa: FBT002
    **get_dataset_kwargs: Any,
) -> OpenMLTask:
    """Download OpenML task for a given task ID.

    Downloads the task representation.

    Use the `download_splits` parameter to control whether the splits are downloaded.
    Moreover, you may pass additional parameter (args or kwargs) that are passed to
    :meth:`openml.datasets.get_dataset`.

    Parameters
    ----------
    task_id : int
        The OpenML task id of the task to download.
    download_splits: bool (default=False)
        Whether to download the splits as well.
    get_dataset_kwargs :
        Args and kwargs can be used pass optional parameters to :meth:`openml.datasets.get_dataset`.

    Returns
    -------
    task: OpenMLTask
    """
    if not isinstance(task_id, int):
            raise TypeError(f"Task id should be integer, is {type(task_id)}")

    task =  api_context.backend.tasks.get(task_id, download_splits=download_splits)
    dataset = get_dataset(task.dataset_id, **get_dataset_kwargs)
    
    if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):
        task.class_labels = dataset.retrieve_class_labels(task.target_name)

    if download_splits and isinstance(task, OpenMLSupervisedTask) and isinstance(api_context.backend.tasks, TasksV1):
        task.download_split()

    return task

def create_task(
    task_type: TaskType,
    dataset_id: int,
    estimation_procedure_id: int,
    target_name: str | None = None,
    evaluation_measure: str | None = None,
    **kwargs: Any,
) -> (
    OpenMLClassificationTask | OpenMLRegressionTask | OpenMLLearningCurveTask | OpenMLClusteringTask
):
    """Create a task based on different given attributes.

    Builds a task object with the function arguments as
    attributes. The type of the task object built is
    determined from the task type id.
    More information on how the arguments (task attributes),
    relate to the different possible tasks can be found in
    the individual task objects at the openml.tasks.task
    module.

    Parameters
    ----------
    task_type : TaskType
        Id of the task type.
    dataset_id : int
        The id of the dataset for the task.
    target_name : str, optional
        The name of the feature used as a target.
        At the moment, only optional for the clustering tasks.
    estimation_procedure_id : int
        The id of the estimation procedure.
    evaluation_measure : str, optional
        The name of the evaluation measure.
    kwargs : dict, optional
        Other task attributes that are not mandatory
        for task upload.

    Returns
    -------
    OpenMLClassificationTask, OpenMLRegressionTask,
    OpenMLLearningCurveTask, OpenMLClusteringTask
    """
    if task_type == TaskType.CLUSTERING:
        task_cls = OpenMLClusteringTask
    elif task_type == TaskType.LEARNING_CURVE:
        task_cls = OpenMLLearningCurveTask  # type: ignore
    elif task_type == TaskType.SUPERVISED_CLASSIFICATION:
        task_cls = OpenMLClassificationTask  # type: ignore
    elif task_type == TaskType.SUPERVISED_REGRESSION:
        task_cls = OpenMLRegressionTask  # type: ignore
    else:
        raise NotImplementedError(f"Task type {task_type:d} not supported.")

    return task_cls(
        task_type_id=task_type,
        task_type="None",  # TODO: refactor to get task type string from ID.
        data_set_id=dataset_id,
        target_name=target_name,  # type: ignore
        estimation_procedure_id=estimation_procedure_id,
        evaluation_measure=evaluation_measure,
        **kwargs,
    )


def delete_task(task_id: int) -> bool:
    """Delete task with id `task_id` from the OpenML server.

    You can only delete tasks which you created and have
    no runs associated with them.

    Parameters
    ----------
    task_id : int
        OpenML id of the task

    Returns
    -------
    bool
        True if the deletion was successful. False otherwise.
    """
    return openml.utils._delete_entity("task", task_id)
