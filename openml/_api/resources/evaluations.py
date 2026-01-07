from __future__ import annotations

import json

import xmltodict

from openml._api.resources.base import EvaluationsAPI
from openml.evaluations.evaluation import OpenMLEvaluation


class EvaluationsV1(EvaluationsAPI):
    """V1 API implementation for evaluations.
    Fetches evaluations from the v1 XML API endpoint.
    """

    def list(self, api_call: str) -> list[OpenMLEvaluation]:
        """Fetch and list evaluations from the OpenML API.

        Makes an API call to retrieve evaluation results, parses the XML response,
        and converts it into OpenMLEvaluation objects.

        Parameters
        ----------
        api_call : str
            The API endpoint path (without base URL) to call for evaluations.
            Example: "evaluation/list/function/predictive_accuracy/limit/10

        Returns
        -------
        list[OpenMLEvaluation]
            A list of OpenMLEvaluation objects containing the parsed evaluations.

        Raises
        ------
        ValueError
            If the XML response does not contain the expected structure.
        AssertionError
            If the evaluation data is not in list format as expected.

        Notes
        -----
        This method performs two API calls:
        1. Fetches evaluation data from the specified endpoint
        2. Fetches user information for all uploaders in the evaluation data

        The user information is used to map uploader IDs to usernames.
        """
        eval_response = self._http.get(api_call)
        xml_content = eval_response.text

        evals_dict = xmltodict.parse(xml_content, force_list=("oml:evaluation",))
        # Minimalistic check if the XML is useful
        if "oml:evaluations" not in evals_dict:
            raise ValueError(
                "Error in return XML, does not contain " f'"oml:evaluations": {evals_dict!s}',
            )

        assert isinstance(evals_dict["oml:evaluations"]["oml:evaluation"], list), type(
            evals_dict["oml:evaluations"]["oml:evaluation"],
        )

        uploader_ids = list(
            {eval_["oml:uploader"] for eval_ in evals_dict["oml:evaluations"]["oml:evaluation"]},
        )
        api_users = "user/list/user_id/" + ",".join(uploader_ids)
        user_response = self._http.get(api_users)
        xml_content_user = user_response.text

        users = xmltodict.parse(xml_content_user, force_list=("oml:user",))
        user_dict = {
            user["oml:id"]: user["oml:username"] for user in users["oml:users"]["oml:user"]
        }

        evals = []
        for eval_ in evals_dict["oml:evaluations"]["oml:evaluation"]:
            run_id = int(eval_["oml:run_id"])
            value = float(eval_["oml:value"]) if "oml:value" in eval_ else None
            values = json.loads(eval_["oml:values"]) if eval_.get("oml:values", None) else None
            array_data = eval_.get("oml:array_data")

            evals.append(
                OpenMLEvaluation(
                    run_id=run_id,
                    task_id=int(eval_["oml:task_id"]),
                    setup_id=int(eval_["oml:setup_id"]),
                    flow_id=int(eval_["oml:flow_id"]),
                    flow_name=eval_["oml:flow_name"],
                    data_id=int(eval_["oml:data_id"]),
                    data_name=eval_["oml:data_name"],
                    function=eval_["oml:function"],
                    upload_time=eval_["oml:upload_time"],
                    uploader=int(eval_["oml:uploader"]),
                    uploader_name=user_dict[eval_["oml:uploader"]],
                    value=value,
                    values=values,
                    array_data=array_data,
                )
            )

        return evals


class EvaluationsV2(EvaluationsAPI):
    """V2 API implementation for evaluations.
    Fetches evaluations from the v2 json API endpoint.
    """

    def list(self, api_call: str) -> list[OpenMLEvaluation]:
        """Fetch and list evaluations from the OpenML API.

        Makes an API call to retrieve evaluation results, parses the json response,
        and converts it into OpenMLEvaluation objects.

        Parameters
        ----------
        api_call : str
            The API endpoint path (without base URL) to call for evaluations.
            Example: "evaluation/list/function/predictive_accuracy/limit/10

        Returns
        -------
        list[OpenMLEvaluation]
            A list of OpenMLEvaluation objects containing the parsed evaluations.

        Raises
        ------
        NotImplementedError

        Notes
        -----
        This method performs two API calls:
        1. Fetches evaluation data from the specified endpoint
        2. Fetches user information for all uploaders in the evaluation data

        The user information is used to map uploader IDs to usernames.
        """
        raise NotImplementedError("V2 API implementation is not yet available")
