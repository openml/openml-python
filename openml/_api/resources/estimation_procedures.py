from __future__ import annotations

import xmltodict

from openml._api.resources.base import EstimationProceduresAPI


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
