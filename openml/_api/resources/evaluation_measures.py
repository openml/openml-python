from __future__ import annotations

import xmltodict

from openml._api.resources.base import EvaluationMeasuresAPI, ResourceV1, ResourceV2


class EvaluationMeasuresV1(ResourceV1, EvaluationMeasuresAPI):
    """V1 API implementation for evaluation measures.

    Fetches evaluation measures from the v1 XML API endpoint.
    """

    def list(self) -> list[str]:
        """List all evaluation measures available on OpenML.

        Returns
        -------
        list[str]
            A list of evaluation measure names.
        """
        path = "evaluationmeasure/list"
        response = self._http.get(path)
        xml_content = response.text

        qualities = xmltodict.parse(xml_content, force_list=("oml:measures"))
        # Minimalistic check if the XML is useful
        if "oml:evaluation_measures" not in qualities:
            raise ValueError('Error in return XML, does not contain "oml:evaluation_measures"')

        if not isinstance(
            qualities["oml:evaluation_measures"]["oml:measures"][0]["oml:measure"], list
        ):
            raise TypeError('Error in return XML, does not contain "oml:measure" as a list')

        return qualities["oml:evaluation_measures"]["oml:measures"][0]["oml:measure"]


class EvaluationMeasuresV2(ResourceV2, EvaluationMeasuresAPI):
    """V2 API implementation for evaluation measures.

    Fetches evaluation measures from the v2 JSON API endpoint.
    """

    def list(self) -> list[str]:
        """List all evaluation measures available on OpenML.

        Returns
        -------
        list[str]
            A list of evaluation measure names.
        """
        path = "evaluationmeasure/list"
        response = self._http.get(path)
        data = response.json()

        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")

        return data
