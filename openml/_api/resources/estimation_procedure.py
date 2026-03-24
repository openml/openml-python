from __future__ import annotations

import builtins
import warnings

import xmltodict

from openml.estimation_procedures.estimation_procedure import OpenMLEstimationProcedure
from openml.tasks.task import TaskType

from .base import EstimationProcedureAPI, ResourceV1API, ResourceV2API


class EstimationProcedureV1API(ResourceV1API, EstimationProcedureAPI):
    """V1 API implementation for estimation procedures.

    Fetches estimation procedures from the v1 XML API endpoint.
    """

    def list(self) -> builtins.list[OpenMLEstimationProcedure]:
        """Return a list of all estimation procedures which are on OpenML.

        Returns
        -------
        procedures : list
            A list of all estimation procedures. Every procedure is represented by
            a dictionary containing the following information: id, task type id,
            name, type, repeats, folds, stratified.
        """
        path = "estimationprocedure/list"
        response = self._http.get(path)
        xml_content = response.text

        procs_dict = xmltodict.parse(xml_content)

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

        procs: builtins.list[OpenMLEstimationProcedure] = []
        for proc_ in procs_dict["oml:estimationprocedures"]["oml:estimationprocedure"]:
            task_type_int = int(proc_["oml:ttid"])
            try:
                task_type_id = TaskType(task_type_int)
                procs.append(
                    OpenMLEstimationProcedure(
                        id=int(proc_["oml:id"]),
                        task_type_id=task_type_id,
                        name=proc_["oml:name"],
                        type=proc_["oml:type"],
                    )
                )
            except ValueError as e:
                warnings.warn(
                    f"Could not create task type id for {task_type_int} due to error {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return procs


class EstimationProcedureV2API(ResourceV2API, EstimationProcedureAPI):
    """V2 API implementation for estimation procedures.

    Fetches estimation procedures from the v2 JSON API endpoint.
    """

    def list(self) -> builtins.list[OpenMLEstimationProcedure]:
        self._not_supported(method="get_details")
