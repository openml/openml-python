from __future__ import annotations

import builtins
import warnings
from typing import Any

import pandas as pd
import xmltodict

from openml._api.resources.base import TasksAPI
from openml.tasks.task import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLTask,
    TaskType,
)

TASKS_CACHE_DIR_NAME = "tasks"


class TasksV1(TasksAPI):
    def get(self, task_id: int) -> OpenMLTask:
        """Download OpenML task for a given task ID.

        Downloads the task representation.

        Parameters
        ----------
        task_id : int
            The OpenML task id of the task to download.
        get_dataset_kwargs :
            Args and kwargs can be used pass optional parameters to
            :meth:`openml.datasets.get_dataset`.

        Returns
        -------
        task: OpenMLTask
        """
        if not isinstance(task_id, int):
            raise TypeError(f"Task id should be integer, is {type(task_id)}")

        response = self._http.get(f"task/{task_id}")
        return self._create_task_from_xml(response.text)

    def _create_task_from_xml(self, xml: str) -> OpenMLTask:
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
            common_kwargs["target_name"] = inputs["source_data"]["oml:data_set"][
                "oml:target_feature"
            ]
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
            raise NotImplementedError(f"Task type {common_kwargs['task_type']} not supported.")
        return cls(**common_kwargs)  # type: ignore

    def list(
        self,
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

        return self._fetch_tasks_df(api_call=api_call)

    def _fetch_tasks_df(self, api_call: str) -> pd.DataFrame:  # noqa: C901, PLR0912
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
        xml_string = self._http.get(api_call).text

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
        procs = self._get_estimation_procedure_list()
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
                    warnings.warn(
                        f"Could not find key {e} in {task_}!", RuntimeWarning, stacklevel=2
                    )

        return pd.DataFrame.from_dict(tasks, orient="index")

    def _get_estimation_procedure_list(self) -> builtins.list[dict[str, Any]]:
        """Return a list of all estimation procedures which are on OpenML.

        Returns
        -------
        procedures : list
            A list of all estimation procedures. Every procedure is represented by
            a dictionary containing the following information: id, task type id,
            name, type, repeats, folds, stratified.
        """
        url_suffix = "estimationprocedure/list"
        xml_string = self._http.get(url_suffix).text

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


class TasksV2(TasksAPI):
    def get(self, task_id: int) -> OpenMLTask:
        response = self._http.get(f"tasks/{task_id}")
        return self._create_task_from_json(response.json())

    def _create_task_from_json(self, task_json: dict) -> OpenMLTask:
        task_type_id = TaskType(int(task_json["task_type_id"]))

        inputs = {i["name"]: i for i in task_json.get("input", [])}

        source = inputs["source_data"]["data_set"]

        common_kwargs = {
            "task_id": int(task_json["id"]),
            "task_type": task_json["task_type"],
            "task_type_id": task_type_id,
            "data_set_id": int(source["data_set_id"]),
            "evaluation_measure": None,
        }

        if task_type_id in (
            TaskType.SUPERVISED_CLASSIFICATION,
            TaskType.SUPERVISED_REGRESSION,
            TaskType.LEARNING_CURVE,
        ):
            est = inputs.get("estimation_procedure", {}).get("estimation_procedure")

            if est:
                common_kwargs["estimation_procedure_id"] = int(est["id"])
                common_kwargs["estimation_procedure_type"] = est["type"]
                common_kwargs["estimation_parameters"] = {
                    p["name"]: p.get("value") for p in est.get("parameter", [])
                }

            common_kwargs["target_name"] = source.get("target_feature")

        cls = {
            TaskType.SUPERVISED_CLASSIFICATION: OpenMLClassificationTask,
            TaskType.SUPERVISED_REGRESSION: OpenMLRegressionTask,
            TaskType.CLUSTERING: OpenMLClusteringTask,
            TaskType.LEARNING_CURVE: OpenMLLearningCurveTask,
        }[task_type_id]

        return cls(**common_kwargs)  # type: ignore

    def list(
        self,
        limit: int,
        offset: int,
        task_type: TaskType | int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        raise NotImplementedError("Task listing is not available in API v2 yet.")
