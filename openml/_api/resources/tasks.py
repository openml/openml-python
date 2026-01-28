from __future__ import annotations

from typing import TYPE_CHECKING

import xmltodict

from openml._api.resources.base import ResourceV1, ResourceV2, TasksAPI
from openml.tasks.task import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLTask,
    TaskType,
)

if TYPE_CHECKING:
    from requests import Response


class TasksV1(ResourceV1, TasksAPI):
    def get(
        self,
        task_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLTask | tuple[OpenMLTask, Response]:
        path = f"task/{task_id}"
        response = self._http.get(path, use_cache=True)
        xml_content = response.text
        task = self._create_task_from_xml(xml_content)

        if return_response:
            return task, response

        return task

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


class TasksV2(ResourceV2, TasksAPI):
    def get(
        self,
        task_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLTask | tuple[OpenMLTask, Response]:
        raise NotImplementedError(self._get_not_implemented_message("get"))
