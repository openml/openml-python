# License: BSD 3-Clause

from collections import OrderedDict
import io
import re
import os
from typing import Union, Dict, Optional

import pandas as pd
import xmltodict

from ..exceptions import OpenMLCacheException
from ..datasets import get_dataset
from .task import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    TaskType,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    OpenMLTask,
)
import openml.utils
import openml._api_calls


TASKS_CACHE_DIR_NAME = "tasks"


def _get_cached_tasks():
    """Return a dict of all the tasks which are cached locally.
    Returns
    -------
    tasks : OrderedDict
        A dict of all the cached tasks. Each task is an instance of
        OpenMLTask.
    """
    tasks = OrderedDict()

    task_cache_dir = openml.utils._create_cache_directory(TASKS_CACHE_DIR_NAME)
    directory_content = os.listdir(task_cache_dir)
    directory_content.sort()
    # Find all dataset ids for which we have downloaded the dataset
    # description

    for filename in directory_content:
        if not re.match(r"[0-9]*", filename):
            continue

        tid = int(filename)
        tasks[tid] = _get_cached_task(tid)

    return tasks


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

    try:
        with io.open(os.path.join(tid_cache_dir, "task.xml"), encoding="utf8") as fh:
            return _create_task_from_xml(fh.read())
    except (OSError, IOError):
        openml.utils._remove_cache_dir_for_id(TASKS_CACHE_DIR_NAME, tid_cache_dir)
        raise OpenMLCacheException("Task file for tid %d not " "cached" % tid)


def _get_estimation_procedure_list():
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
        raise ValueError("Error in return XML, does not contain tag " "oml:estimationprocedures.")
    elif "@xmlns:oml" not in procs_dict["oml:estimationprocedures"]:
        raise ValueError(
            "Error in return XML, does not contain tag "
            "@xmlns:oml as a child of oml:estimationprocedures."
        )
    elif procs_dict["oml:estimationprocedures"]["@xmlns:oml"] != "http://openml.org/openml":
        raise ValueError(
            "Error in return XML, value of "
            "oml:estimationprocedures/@xmlns:oml is not "
            "http://openml.org/openml, but %s"
            % str(procs_dict["oml:estimationprocedures"]["@xmlns:oml"])
        )

    procs = []
    for proc_ in procs_dict["oml:estimationprocedures"]["oml:estimationprocedure"]:
        procs.append(
            {
                "id": int(proc_["oml:id"]),
                "task_type_id": TaskType(int(proc_["oml:ttid"])),
                "name": proc_["oml:name"],
                "type": proc_["oml:type"],
            }
        )

    return procs


def list_tasks(
    task_type: Optional[TaskType] = None,
    offset: Optional[int] = None,
    size: Optional[int] = None,
    tag: Optional[str] = None,
    output_format: str = "dict",
    **kwargs
) -> Union[Dict, pd.DataFrame]:
    """
    Return a number of tasks having the given tag and task_type

    Parameters
    ----------
    Filter task_type is separated from the other filters because
    it is used as task_type in the task description, but it is named
    type when used as a filter in list tasks call.
    task_type : TaskType, optional
        ID of the task type as detailed `here <https://www.openml.org/search?type=task_type>`_.
        - Supervised classification: 1
        - Supervised regression: 2
        - Learning curve: 3
        - Supervised data stream classification: 4
        - Clustering: 5
        - Machine Learning Challenge: 6
        - Survival Analysis: 7
        - Subgroup Discovery: 8
    offset : int, optional
        the number of tasks to skip, starting from the first
    size : int, optional
        the maximum number of tasks to show
    tag : str, optional
        the tag to include
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    kwargs: dict, optional
        Legal filter operators: data_tag, status, data_id, data_name,
        number_instances, number_features,
        number_classes, number_missing_values.

    Returns
    -------
    dict
        All tasks having the given task_type and the give tag. Every task is
        represented by a dictionary containing the following information:
        task id, dataset id, task_type and status. If qualities are calculated
        for the associated dataset, some of these are also returned.
    dataframe
        All tasks having the given task_type and the give tag. Every task is
        represented by a row in the data frame containing the following information
        as columns: task id, dataset id, task_type and status. If qualities are
        calculated for the associated dataset, some of these are also returned.
    """
    if output_format not in ["dataframe", "dict"]:
        raise ValueError(
            "Invalid output format selected. " "Only 'dict' or 'dataframe' applicable."
        )
    return openml.utils._list_all(
        output_format=output_format,
        listing_call=_list_tasks,
        task_type=task_type,
        offset=offset,
        size=size,
        tag=tag,
        **kwargs
    )


def _list_tasks(task_type=None, output_format="dict", **kwargs):
    """
    Perform the api call to return a number of tasks having the given filters.
    Parameters
    ----------
    Filter task_type is separated from the other filters because
    it is used as task_type in the task description, but it is named
    type when used as a filter in list tasks call.
    task_type : TaskType, optional
        ID of the task type as detailed
        `here <https://www.openml.org/search?type=task_type>`_.
        - Supervised classification: 1
        - Supervised regression: 2
        - Learning curve: 3
        - Supervised data stream classification: 4
        - Clustering: 5
        - Machine Learning Challenge: 6
        - Survival Analysis: 7
        - Subgroup Discovery: 8
    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame
    kwargs: dict, optional
        Legal filter operators: tag, task_id (list), data_tag, status, limit,
        offset, data_id, data_name, number_instances, number_features,
        number_classes, number_missing_values.

    Returns
    -------
    dict or dataframe
    """
    api_call = "task/list"
    if task_type is not None:
        api_call += "/type/%d" % task_type.value
    if kwargs is not None:
        for operator, value in kwargs.items():
            if operator == "task_id":
                value = ",".join([str(int(i)) for i in value])
            api_call += "/%s/%s" % (operator, value)
    return __list_tasks(api_call=api_call, output_format=output_format)


def __list_tasks(api_call, output_format="dict"):
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    tasks_dict = xmltodict.parse(xml_string, force_list=("oml:task", "oml:input"))
    # Minimalistic check if the XML is useful
    if "oml:tasks" not in tasks_dict:
        raise ValueError('Error in return XML, does not contain "oml:runs": %s' % str(tasks_dict))
    elif "@xmlns:oml" not in tasks_dict["oml:tasks"]:
        raise ValueError(
            "Error in return XML, does not contain " '"oml:runs"/@xmlns:oml: %s' % str(tasks_dict)
        )
    elif tasks_dict["oml:tasks"]["@xmlns:oml"] != "http://openml.org/openml":
        raise ValueError(
            "Error in return XML, value of  "
            '"oml:runs"/@xmlns:oml is not '
            '"http://openml.org/openml": %s' % str(tasks_dict)
        )

    assert type(tasks_dict["oml:tasks"]["oml:task"]) == list, type(tasks_dict["oml:tasks"])

    tasks = dict()
    procs = _get_estimation_procedure_list()
    proc_dict = dict((x["id"], x) for x in procs)

    for task_ in tasks_dict["oml:tasks"]["oml:task"]:
        tid = None
        try:
            tid = int(task_["oml:task_id"])
            task = {
                "tid": tid,
                "ttid": TaskType(int(task_["oml:task_type_id"])),
                "did": int(task_["oml:did"]),
                "name": task_["oml:name"],
                "task_type": task_["oml:task_type"],
                "status": task_["oml:status"],
            }

            # Other task inputs
            for input in task_.get("oml:input", list()):
                if input["@name"] == "estimation_procedure":
                    task[input["@name"]] = proc_dict[int(input["#text"])]["name"]
                else:
                    value = input.get("#text")
                    task[input["@name"]] = value

            # The number of qualities can range from 0 to infinity
            for quality in task_.get("oml:quality", list()):
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
                raise KeyError("Invalid xml for task %d: %s\nFrom %s" % (tid, e, task_))
            else:
                raise KeyError("Could not find key %s in %s!" % (e, task_))

    if output_format == "dataframe":
        tasks = pd.DataFrame.from_dict(tasks, orient="index")

    return tasks


def get_tasks(task_ids, download_data=True):
    """Download tasks.

    This function iterates :meth:`openml.tasks.get_task`.

    Parameters
    ----------
    task_ids : iterable
        Integers/Strings representing task ids.
    download_data : bool
        Option to trigger download of data along with the meta data.

    Returns
    -------
    list
    """
    tasks = []
    for task_id in task_ids:
        tasks.append(get_task(task_id, download_data))
    return tasks


@openml.utils.thread_safe_if_oslo_installed
def get_task(task_id: int, download_data: bool = True) -> OpenMLTask:
    """Download OpenML task for a given task ID.

    Downloads the task representation, while the data splits can be
    downloaded optionally based on the additional parameter. Else,
    splits will either way be downloaded when the task is being used.

    Parameters
    ----------
    task_id : int or str
        The OpenML task id.
    download_data : bool
        Option to trigger download of data along with the meta data.

    Returns
    -------
    task
    """
    try:
        task_id = int(task_id)
    except (ValueError, TypeError):
        raise ValueError("Dataset ID is neither an Integer nor can be " "cast to an Integer.")

    tid_cache_dir = openml.utils._create_cache_directory_for_id(TASKS_CACHE_DIR_NAME, task_id,)

    try:
        task = _get_task_description(task_id)
        dataset = get_dataset(task.dataset_id, download_data)
        # List of class labels availaible in dataset description
        # Including class labels as part of task meta data handles
        #   the case where data download was initially disabled
        if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):
            task.class_labels = dataset.retrieve_class_labels(task.target_name)
        # Clustering tasks do not have class labels
        # and do not offer download_split
        if download_data:
            if isinstance(task, OpenMLSupervisedTask):
                task.download_split()
    except Exception as e:
        openml.utils._remove_cache_dir_for_id(
            TASKS_CACHE_DIR_NAME, tid_cache_dir,
        )
        raise e

    return task


def _get_task_description(task_id):

    try:
        return _get_cached_task(task_id)
    except OpenMLCacheException:
        xml_file = os.path.join(
            openml.utils._create_cache_directory_for_id(TASKS_CACHE_DIR_NAME, task_id,), "task.xml",
        )
        task_xml = openml._api_calls._perform_api_call("task/%d" % task_id, "get")

        with io.open(xml_file, "w", encoding="utf8") as fh:
            fh.write(task_xml)
        return _create_task_from_xml(task_xml)


def _create_task_from_xml(xml):
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
    estimation_parameters = dict()
    inputs = dict()
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
        raise NotImplementedError("Task type %s not supported." % common_kwargs["task_type"])
    return cls(**common_kwargs)


def create_task(
    task_type: TaskType,
    dataset_id: int,
    estimation_procedure_id: int,
    target_name: Optional[str] = None,
    evaluation_measure: Optional[str] = None,
    **kwargs
) -> Union[
    OpenMLClassificationTask, OpenMLRegressionTask, OpenMLLearningCurveTask, OpenMLClusteringTask
]:
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
    task_cls = {
        TaskType.SUPERVISED_CLASSIFICATION: OpenMLClassificationTask,
        TaskType.SUPERVISED_REGRESSION: OpenMLRegressionTask,
        TaskType.CLUSTERING: OpenMLClusteringTask,
        TaskType.LEARNING_CURVE: OpenMLLearningCurveTask,
    }.get(task_type)

    if task_cls is None:
        raise NotImplementedError("Task type {0:d} not supported.".format(task_type))
    else:
        return task_cls(
            task_type_id=task_type,
            task_type=None,
            data_set_id=dataset_id,
            target_name=target_name,
            estimation_procedure_id=estimation_procedure_id,
            evaluation_measure=evaluation_measure,
            **kwargs
        )
