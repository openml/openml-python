# License: BSD 3-Clause
from __future__ import annotations

import os
import re
import warnings
from functools import partial
from typing import Any

import pandas as pd
import xmltodict

import openml._api_calls
import openml.utils
from openml.datasets import get_dataset
from openml.exceptions import OpenMLCacheException

from .task import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    OpenMLTask,
    TaskType,
)

TASKS_CACHE_DIR_NAME = "tasks"


def _get_cached_tasks() -> dict[int, OpenMLTask]:
    """Return a dict of all the tasks which are cached locally.

    Returns
    -------
    tasks : OrderedDict
        A dict of all the cached tasks. Each task is an instance of
        OpenMLTask.
    """
    task_cache_dir = openml.utils._create_cache_directory(TASKS_CACHE_DIR_NAME)
    directory_content = os.listdir(task_cache_dir)  # noqa: PTH208
    directory_content.sort()

    # Find all dataset ids for which we have downloaded the dataset
    # description
    tids = (int(did) for did in directory_content if re.match(r"[0-9]*", did))
    return {tid: _get_cached_task(tid) for tid in tids}


def _get_cached_task(tid: int) -> OpenMLTask:
    """Return a cached task based on the given id.

    Parameters
    ----------
    tid : int
        Id of the task.

    Returns
    -------
    OpenMLTask
    """
    tid_cache_dir = openml.utils._create_cache_directory_for_id(TASKS_CACHE_DIR_NAME, tid)

    task_xml_path = tid_cache_dir / "task.xml"
    try:
        with task_xml_path.open(encoding="utf8") as fh:
            return _create_task_from_xml(fh.read())
    except OSError as e:
        openml.utils._remove_cache_dir_for_id(TASKS_CACHE_DIR_NAME, tid_cache_dir)
        raise OpenMLCacheException(f"Task file for tid {tid} not cached") from e


def _get_estimation_procedure_list() -> list[dict[str, Any]]:
    """Return a list of all estimation procedures which are on OpenML.

    Returns
    -------
    procedures : list
        A list of all estimation procedures. Every procedure is represented by
        a dictionary containing the following information: id, task type id,
        name, type, repeats, folds, stratified.
    """
    url_suffix = "estimationprocedure/list"
    xml_string = openml._api_calls._perform_api_call(url_suffix, "get")

    procs_dict = xmltodict.parse(xml_string)
    # Minimalistic check if the XML is useful
    if "oml:estimationprocedures" not in procs_dict:
        raise ValueError("Error in return XML, does not contain tag oml:estimationprocedures.")

    if "@xmlns:oml" not in procs_dict["oml:estimationprocedures"]:
        raise ValueError(
            "Error in return XML, does not contain tag "
            "@xmlns:oml as a child of oml:estimationprocedures.",
        )

    if procs_dict["oml:estimationprocedures"]["@xmlns:oml"] != "http://openml.org/openml":
        raise ValueError(
            "Error in return XML, value of "
            "oml:estimationprocedures/@xmlns:oml is not "
            "http://openml.org/openml, but {}".format(
                str(procs_dict["oml:estimationprocedures"]["@xmlns:oml"])
            ),
        )

    procs: list[dict[str, Any]] = []
    for proc_ in procs_dict["oml:estimationprocedures"]["oml:estimationprocedure"]:
        task_type_int = int(proc_["oml:ttid"])
        try:
            task_type_id = TaskType(task_type_int)
            procs.append(
                {
                    "id": int(proc_["oml:id"]),
                    "task_type_id": task_type_id,
                    "name": proc_["oml:name"],
                    "type": proc_["oml:type"],
                },
            )
        except ValueError as e:
            warnings.warn(
                f"Could not create task type id for {task_type_int} due to error {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    return procs


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
        _list_tasks,
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


def _list_tasks(
    limit: int,
    offset: int,
    task_type: TaskType | int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Perform the api call to return a number of tasks having the given filters.

    Parameters
    ----------
    Filter task_type is separated from the other filters because
    it is used as task_type in the task description, but it is named
    type when used as a filter in list tasks call.
    limit: int
    offset: int
    task_type : TaskType, optional
        Refers to the type of task.
    kwargs: dict, optional
        Legal filter operators: tag, task_id (list), data_tag, status, limit,
        offset, data_id, data_name, number_instances, number_features,
        number_classes, number_missing_values.

    Returns
    -------
    dataframe
    """
    api_call = "task/list"
    if limit is not None:
        api_call += f"/limit/{limit}"
    if offset is not None:
        api_call += f"/offset/{offset}"
    if task_type is not None:
        tvalue = task_type.value if isinstance(task_type, TaskType) else task_type
        api_call += f"/type/{tvalue}"
    if kwargs is not None:
        for operator, value in kwargs.items():
            if value is not None:
                if operator == "task_id":
                    value = ",".join([str(int(i)) for i in value])  # noqa: PLW2901
                api_call += f"/{operator}/{value}"

    return __list_tasks(api_call=api_call)


def __list_tasks(api_call: str) -> pd.DataFrame:  # noqa: C901, PLR0912
    """Returns a Pandas DataFrame with information about OpenML tasks.

    Parameters
    ----------
    api_call : str
        The API call specifying which tasks to return.

    Returns
    -------
        A Pandas DataFrame with information about OpenML tasks.

    Raises
    ------
    ValueError
        If the XML returned by the OpenML API does not contain 'oml:tasks', '@xmlns:oml',
        or has an incorrect value for '@xmlns:oml'.
    KeyError
        If an invalid key is found in the XML for a task.
    """
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    tasks_dict = xmltodict.parse(xml_string, force_list=("oml:task", "oml:input"))
    # Minimalistic check if the XML is useful
    if "oml:tasks" not in tasks_dict:
        raise ValueError(f'Error in return XML, does not contain "oml:runs": {tasks_dict}')

    if "@xmlns:oml" not in tasks_dict["oml:tasks"]:
        raise ValueError(
            f'Error in return XML, does not contain "oml:runs"/@xmlns:oml: {tasks_dict}'
        )

    if tasks_dict["oml:tasks"]["@xmlns:oml"] != "http://openml.org/openml":
        raise ValueError(
            "Error in return XML, value of  "
            '"oml:runs"/@xmlns:oml is not '
            f'"http://openml.org/openml": {tasks_dict!s}',
        )

    assert isinstance(tasks_dict["oml:tasks"]["oml:task"], list), type(tasks_dict["oml:tasks"])

    tasks = {}
    procs = _get_estimation_procedure_list()
    proc_dict = {x["id"]: x for x in procs}

    for task_ in tasks_dict["oml:tasks"]["oml:task"]:
        tid = None
        try:
            tid = int(task_["oml:task_id"])
            task_type_int = int(task_["oml:task_type_id"])
            try:
                task_type_id = TaskType(task_type_int)
            except ValueError as e:
                warnings.warn(
                    f"Could not create task type id for {task_type_int} due to error {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            task = {
                "tid": tid,
                "ttid": task_type_id,
                "did": int(task_["oml:did"]),
                "name": task_["oml:name"],
                "task_type": task_["oml:task_type"],
                "status": task_["oml:status"],
            }

            # Other task inputs
            for _input in task_.get("oml:input", []):
                if _input["@name"] == "estimation_procedure":
                    task[_input["@name"]] = proc_dict[int(_input["#text"])]["name"]
                else:
                    value = _input.get("#text")
                    task[_input["@name"]] = value

            # The number of qualities can range from 0 to infinity
            for quality in task_.get("oml:quality", []):
                if "#text" not in quality:
                    quality_value = 0.0
                else:
                    quality["#text"] = float(quality["#text"])
                    if abs(int(quality["#text"]) - quality["#text"]) < 0.0000001:
                        quality["#text"] = int(quality["#text"])
                    quality_value = quality["#text"]
                task[quality["@name"]] = quality_value
            tasks[tid] = task
        except KeyError as e:
            if tid is not None:
                warnings.warn(
                    f"Invalid xml for task {tid}: {e}\nFrom {task_}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(f"Could not find key {e} in {task_}!", RuntimeWarning, stacklevel=2)

    return pd.DataFrame.from_dict(tasks, orient="index")


def get_tasks(
    task_ids: list[int],
    download_data: bool | None = None,
    download_qualities: bool | None = None,
) -> list[OpenMLTask]:
    """Download tasks.

    This function iterates :meth:`openml.tasks.get_task`.

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
            get_task(task_id, download_data=download_data, download_qualities=download_qualities)
        )
    return tasks


@openml.utils.thread_safe_if_oslo_installed
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

    task_cache_directory = openml.utils._create_cache_directory_for_id(
        TASKS_CACHE_DIR_NAME, task_id
    )
    task_cache_directory_existed = task_cache_directory.exists()
    try:
        task = _get_task_description(task_id)
        dataset = get_dataset(task.dataset_id, **get_dataset_kwargs)
        # List of class labels available in dataset description
        # Including class labels as part of task meta data handles
        #   the case where data download was initially disabled
        if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):
            assert task.target_name is not None, (
                "Supervised tasks must define a target feature before retrieving class labels."
            )
            task.class_labels = dataset.retrieve_class_labels(task.target_name)
        # Clustering tasks do not have class labels
        # and do not offer download_split
        if download_splits and isinstance(task, OpenMLSupervisedTask):
            task.download_split()
    except Exception as e:
        if not task_cache_directory_existed:
            openml.utils._remove_cache_dir_for_id(TASKS_CACHE_DIR_NAME, task_cache_directory)
        raise e

    return task


def _get_task_description(task_id: int) -> OpenMLTask:
    try:
        return _get_cached_task(task_id)
    except OpenMLCacheException:
        _cache_dir = openml.utils._create_cache_directory_for_id(TASKS_CACHE_DIR_NAME, task_id)
        xml_file = _cache_dir / "task.xml"
        task_xml = openml._api_calls._perform_api_call(f"task/{task_id}", "get")

        with xml_file.open("w", encoding="utf8") as fh:
            fh.write(task_xml)
        return _create_task_from_xml(task_xml)


def _create_task_from_xml(xml: str) -> OpenMLTask:
    """Create a task given a xml string.

    Parameters
    ----------
    xml : string
        Task xml representation.

    Returns
    -------
    OpenMLTask
    """
    dic = xmltodict.parse(xml)["oml:task"]
    estimation_parameters = {}
    inputs = {}
    # Due to the unordered structure we obtain, we first have to extract
    # the possible keys of oml:input; dic["oml:input"] is a list of
    # OrderedDicts

    # Check if there is a list of inputs
    if isinstance(dic["oml:input"], list):
        for input_ in dic["oml:input"]:
            name = input_["@name"]
            inputs[name] = input_
    # Single input case
    elif isinstance(dic["oml:input"], dict):
        name = dic["oml:input"]["@name"]
        inputs[name] = dic["oml:input"]

    evaluation_measures = None
    if "evaluation_measures" in inputs:
        evaluation_measures = inputs["evaluation_measures"]["oml:evaluation_measures"][
            "oml:evaluation_measure"
        ]

    task_type = TaskType(int(dic["oml:task_type_id"]))
    common_kwargs = {
        "task_id": dic["oml:task_id"],
        "task_type": dic["oml:task_type"],
        "task_type_id": task_type,
        "data_set_id": inputs["source_data"]["oml:data_set"]["oml:data_set_id"],
        "evaluation_measure": evaluation_measures,
    }
    # TODO: add OpenMLClusteringTask?
    if task_type in (
        TaskType.SUPERVISED_CLASSIFICATION,
        TaskType.SUPERVISED_REGRESSION,
        TaskType.LEARNING_CURVE,
    ):
        # Convert some more parameters
        for parameter in inputs["estimation_procedure"]["oml:estimation_procedure"][
            "oml:parameter"
        ]:
            name = parameter["@name"]
            text = parameter.get("#text", "")
            estimation_parameters[name] = text

        common_kwargs["estimation_procedure_type"] = inputs["estimation_procedure"][
            "oml:estimation_procedure"
        ]["oml:type"]
        common_kwargs["estimation_procedure_id"] = int(
            inputs["estimation_procedure"]["oml:estimation_procedure"]["oml:id"]
        )

        common_kwargs["estimation_parameters"] = estimation_parameters
        common_kwargs["target_name"] = inputs["source_data"]["oml:data_set"]["oml:target_feature"]
        common_kwargs["data_splits_url"] = inputs["estimation_procedure"][
            "oml:estimation_procedure"
        ]["oml:data_splits_url"]

    cls = {
        TaskType.SUPERVISED_CLASSIFICATION: OpenMLClassificationTask,
        TaskType.SUPERVISED_REGRESSION: OpenMLRegressionTask,
        TaskType.CLUSTERING: OpenMLClusteringTask,
        TaskType.LEARNING_CURVE: OpenMLLearningCurveTask,
    }.get(task_type)
    if cls is None:
        raise NotImplementedError(
            f"Task type '{common_kwargs['task_type']}' is not supported. "
            f"Supported task types: SUPERVISED_CLASSIFICATION,"
            f"SUPERVISED_REGRESSION, CLUSTERING, LEARNING_CURVE."
            f"Please check the OpenML documentation for available task types."
        )
    return cls(**common_kwargs)  # type: ignore


# TODO(eddiebergman): overload on `task_type`
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
        raise NotImplementedError(
            f"Task type ID {task_type:d} is not supported. "
            f"Supported task type IDs: {TaskType.SUPERVISED_CLASSIFICATION.value},"
            f"{TaskType.SUPERVISED_REGRESSION.value}, "
            f"{TaskType.CLUSTERING.value}, {TaskType.LEARNING_CURVE.value}. "
            f"Please refer to the TaskType enum for valid task type identifiers."
        )

    return task_cls(
        task_id=None,
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
