from __future__ import annotations

import warnings
from typing import Any

import xmltodict

from openml._api.resources.base import EstimationProceduresAPI
from openml.tasks.task import TaskType


class EstimationProceduresV1(EstimationProceduresAPI):
    """V1 API implementation for estimation procedures.

    Fetches estimation procedures from the v1 XML API endpoint.
    """

    def list(self) -> list[str]:
        """List the names of all estimation procedures available on OpenML.

        Returns
        -------
        list[str]
        """
        path = "estimationprocedure/list"
        response = self._http.get(path)
        xml_content = response.text

        api_results = xmltodict.parse(xml_content)

        # Minimalistic check if the XML is useful
        if "oml:estimationprocedures" not in api_results:
            raise ValueError('Error in return XML, does not contain "oml:estimationprocedures"')

        if "oml:estimationprocedure" not in api_results["oml:estimationprocedures"]:
            raise ValueError('Error in return XML, does not contain "oml:estimationprocedure"')

        if not isinstance(api_results["oml:estimationprocedures"]["oml:estimationprocedure"], list):
            raise TypeError(
                'Error in return XML, does not contain "oml:estimationprocedure" as a list'
            )

        return [
            prod["oml:name"]
            for prod in api_results["oml:estimationprocedures"]["oml:estimationprocedure"]
        ]

    def _get_details(self) -> list[dict[str, Any]]:
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


class EstimationProceduresV2(EstimationProceduresAPI):
    """V2 API implementation for estimation procedures.

    Fetches estimation procedures from the v2 JSON API endpoint.
    """

    def list(self) -> list[str]:
        """List the names of all estimation procedures available on OpenML.

        Returns
        -------
        list[str]
        """
        path = "estimationprocedure/list"
        response = self._http.get(path)
        list_of_prod_dicts = response.json()

        if not isinstance(list_of_prod_dicts, list):
            raise TypeError(f"Expected list response, got {type(list_of_prod_dicts)}")

        return [prod["name"] for prod in list_of_prod_dicts]

    def _get_details(self) -> list[dict[str, Any]]:
        raise NotImplementedError("V2 API implementation is not yet available")
