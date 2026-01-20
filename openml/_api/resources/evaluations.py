from __future__ import annotations

import json
from typing import Any

import xmltodict

from openml._api.resources.base import EvaluationsAPI
from openml.evaluations import OpenMLEvaluation


class EvaluationsV1(EvaluationsAPI):
    """V1 API implementation for evaluations.
    Fetches evaluations from the v1 XML API endpoint.
    """

    def list(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        function: str,
        tasks: list | None = None,
        setups: list | None = None,
        flows: list | None = None,
        runs: list | None = None,
        uploaders: list | None = None,
        study: int | None = None,
        sort_order: str | None = None,
        **kwargs: Any,
    ) -> list[OpenMLEvaluation]:
        """Retrieve evaluations from the OpenML v1 XML API.

        This method builds an evaluation query URL based on the provided
        filters, sends a request to the OpenML v1 endpoint, parses the XML
        response into a dictionary, and enriches the result with uploader
        usernames.

        Parameters
        ----------
        The arguments that are lists are separated from the single value
        ones which are put into the kwargs.

        limit : int
            the number of evaluations to return
        offset : int
            the number of evaluations to skip, starting from the first
        function : str
            the evaluation function. e.g., predictive_accuracy

        tasks : list[int,str], optional
            the list of task IDs
        setups: list[int,str], optional
            the list of setup IDs
        flows : list[int,str], optional
            the list of flow IDs
        runs :list[int,str], optional
            the list of run IDs
        uploaders : list[int,str], optional
            the list of uploader IDs

        study : int, optional

        kwargs: dict, optional
            Legal filter operators: tag, per_fold

        sort_order : str, optional
            order of sorting evaluations, ascending ("asc") or descending ("desc")

        Returns
        -------
        list of OpenMLEvaluation objects

        Notes
        -----
        This method performs two API calls:
        1. Fetches evaluation data from the specified endpoint
        2. Fetches user information for all uploaders in the evaluation data

        The user information is used to map uploader IDs to usernames.
        """
        api_call = self._build_url(
            limit,
            offset,
            function=function,
            tasks=tasks,
            setups=setups,
            flows=flows,
            runs=runs,
            uploaders=uploaders,
            study=study,
            sort_order=sort_order,
            **kwargs,
        )

        eval_response = self._http.get(api_call)
        xml_content = eval_response.text

        return self._parse_list_xml(xml_content)

    def _build_url(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        function: str,
        tasks: list | None = None,
        setups: list | None = None,
        flows: list | None = None,
        runs: list | None = None,
        uploaders: list | None = None,
        study: int | None = None,
        sort_order: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Construct an OpenML evaluation API URL with filtering parameters.

        Parameters
        ----------
        The arguments that are lists are separated from the single value
        ones which are put into the kwargs.

        limit : int
            the number of evaluations to return
        offset : int
            the number of evaluations to skip, starting from the first
        function : str
            the evaluation function. e.g., predictive_accuracy

        tasks : list[int,str], optional
            the list of task IDs
        setups: list[int,str], optional
            the list of setup IDs
        flows : list[int,str], optional
            the list of flow IDs
        runs :list[int,str], optional
            the list of run IDs
        uploaders : list[int,str], optional
            the list of uploader IDs

        study : int, optional

        kwargs: dict, optional
            Legal filter operators: tag, per_fold

        sort_order : str, optional
            order of sorting evaluations, ascending ("asc") or descending ("desc")

        Returns
        -------
        str
            A relative API path suitable for an OpenML HTTP request.
        """
        api_call = f"evaluation/list/function/{function}"
        if limit is not None:
            api_call += f"/limit/{limit}"
        if offset is not None:
            api_call += f"/offset/{offset}"
        if kwargs is not None:
            for operator, value in kwargs.items():
                if value is not None:
                    api_call += f"/{operator}/{value}"
        if tasks is not None:
            api_call += f"/task/{','.join([str(int(i)) for i in tasks])}"
        if setups is not None:
            api_call += f"/setup/{','.join([str(int(i)) for i in setups])}"
        if flows is not None:
            api_call += f"/flow/{','.join([str(int(i)) for i in flows])}"
        if runs is not None:
            api_call += f"/run/{','.join([str(int(i)) for i in runs])}"
        if uploaders is not None:
            api_call += f"/uploader/{','.join([str(int(i)) for i in uploaders])}"
        if study is not None:
            api_call += f"/study/{study}"
        if sort_order is not None:
            api_call += f"/sort_order/{sort_order}"

        return api_call

    def _parse_list_xml(self, xml_content: str) -> list[OpenMLEvaluation]:
        """Helper function to parse API calls which are lists of runs"""
        evals_dict: dict[str, Any] = xmltodict.parse(xml_content, force_list=("oml:evaluation",))
        # Minimalistic check if the XML is useful
        if "oml:evaluations" not in evals_dict:
            raise ValueError(
                f'Error in return XML, does not contain "oml:evaluations": {evals_dict!s}',
            )

        assert isinstance(evals_dict["oml:evaluations"]["oml:evaluation"], list), (
            "Expected 'oml:evaluation' to be a list, but got "
            f"{type(evals_dict['oml:evaluations']['oml:evaluation']).__name__}. "
        )

        uploader_ids = list(
            {eval_["oml:uploader"] for eval_ in evals_dict["oml:evaluations"]["oml:evaluation"]},
        )
        user_dict = self.get_users(uploader_ids)

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

    def get_users(self, uploader_ids: list[str]) -> dict:
        """
        Retrieve usernames for a list of OpenML user IDs.

        Parameters
        ----------
        uploader_ids : list[str]
            List of OpenML user IDs.

        Returns
        -------
        dict
            A mapping from user ID (str) to username (str).
        """
        api_users = "user/list/user_id/" + ",".join(uploader_ids)
        user_response = self._http.get(api_users)
        xml_content_user = user_response.text

        users = xmltodict.parse(xml_content_user, force_list=("oml:user",))
        return {user["oml:id"]: user["oml:username"] for user in users["oml:users"]["oml:user"]}


class EvaluationsV2(EvaluationsAPI):
    """V2 API implementation for evaluations.
    Fetches evaluations from the v2 json API endpoint.
    """

    def list(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        function: str,
        tasks: list | None = None,
        setups: list | None = None,
        flows: list | None = None,
        runs: list | None = None,
        uploaders: list | None = None,
        study: int | None = None,
        sort_order: str | None = None,
        **kwargs: Any,
    ) -> list[OpenMLEvaluation]:
        """
        Retrieve evaluation results from the OpenML v2 JSON API.

        Notes
        -----
        This method is not yet implemented.
        """
        raise NotImplementedError("V2 API implementation is not yet available")

    def get_users(self, uploader_ids: list[str]) -> dict:
        """
        Retrieve usernames for a list of OpenML user IDs using the v2 API.

        Notes
        -----
        This method is not yet implemented.
        """
        raise NotImplementedError("V2 API implementation is not yet available")
