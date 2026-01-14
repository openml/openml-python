from __future__ import annotations

from typing import Any

import xmltodict

from openml._api.resources.base import EvaluationsAPI


class EvaluationsV1(EvaluationsAPI):
    """V1 API implementation for evaluations.
    Fetches evaluations from the v1 XML API endpoint.
    """

    def list(
        self,
        limit: int,
        offset: int,
        function: str,
        **kwargs: Any,
    ) -> dict:
        """Retrieve evaluations from the OpenML v1 XML API.

        This method builds an evaluation query URL based on the provided
        filters, sends a request to the OpenML v1 endpoint, parses the XML
        response into a dictionary, and enriches the result with uploader
        usernames.

        Parameters
        ----------
        limit : int
            Maximum number of evaluations to return.
        offset : int
            Offset for pagination.
        function : str
            the evaluation function. e.g., predictive_accuracy
        **kwargs
            Optional filters supported by the OpenML evaluation API, such as:
            - tasks
            - setups
            - flows
            - runs
            - uploaders
            - tag
            - study
            - sort_order

        Returns
        -------
        dict
            A dictionary containing:
            - Parsed evaluation data from the XML response
            - A "users" key mapping uploader IDs to usernames

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
        api_call = self._build_url(limit, offset, function, **kwargs)
        eval_response = self._http.get(api_call)
        xml_content = eval_response.text

        evals_dict: dict[str, Any] = xmltodict.parse(xml_content, force_list=("oml:evaluation",))
        # Minimalistic check if the XML is useful
        if "oml:evaluations" not in evals_dict:
            raise ValueError(
                "Error in return XML, does not contain " f'"oml:evaluations": {evals_dict!s}',
            )

        assert isinstance(evals_dict["oml:evaluations"]["oml:evaluation"], list), (
            "Expected 'oml:evaluation' to be a list, but got "
            f"{type(evals_dict['oml:evaluations']['oml:evaluation']).__name__}. "
        )

        uploader_ids = list(
            {eval_["oml:uploader"] for eval_ in evals_dict["oml:evaluations"]["oml:evaluation"]},
        )
        user_dict = self.get_users(uploader_ids)
        evals_dict["users"] = user_dict

        return evals_dict

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

    def _build_url(
        self,
        limit: int,
        offset: int,
        function: str,
        **kwargs: Any,
    ) -> str:
        """
        Construct an OpenML evaluation API URL with filtering parameters.

        Parameters
        ----------
        limit : int
            Maximum number of evaluations to return.
        offset : int
            Offset for pagination.
        function : str
            the evaluation function. e.g., predictive_accuracy
        **kwargs
            Evaluation filters such as task IDs, flow IDs,
            uploader IDs, study name, and sorting options.

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

        # List-based filters
        list_filters = {
            "task": kwargs.get("tasks"),
            "setup": kwargs.get("setups"),
            "flow": kwargs.get("flows"),
            "run": kwargs.get("runs"),
            "uploader": kwargs.get("uploaders"),
        }

        for name, values in list_filters.items():
            if values is not None:
                api_call += f"/{name}/" + ",".join(str(int(v)) for v in values)

        # Single-value filters
        if kwargs.get("study") is not None:
            api_call += f"/study/{kwargs['study']}"

        if kwargs.get("sort_order") is not None:
            api_call += f"/sort_order/{kwargs['sort_order']}"

        # Extra filters (tag, per_fold, future-proof)
        for key in ("tag", "per_fold"):
            value = kwargs.get(key)
            if value is not None:
                api_call += f"/{key}/{value}"

        return api_call


class EvaluationsV2(EvaluationsAPI):
    """V2 API implementation for evaluations.
    Fetches evaluations from the v2 json API endpoint.
    """

    def list(
        self,
        limit: int,
        offset: int,
        function: str,
        **kwargs: Any,
    ) -> dict:
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
