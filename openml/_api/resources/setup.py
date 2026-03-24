from __future__ import annotations

import builtins
from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import xmltodict

from openml.setups.setup import OpenMLParameter, OpenMLSetup

from .base import ResourceV1API, ResourceV2API, SetupAPI

if TYPE_CHECKING:
    from openml.flows.flow import OpenMLFlow


class SetupV1API(ResourceV1API, SetupAPI):
    """V1 XML API implementation for setups."""

    def list(
        self,
        limit: int,
        offset: int,
        *,
        setup: Iterable[int] | None = None,
        flow: int | None = None,
        tag: str | None = None,
    ) -> builtins.list[OpenMLSetup]:
        """Perform API call `/setup/list/{filters}`

        Parameters
        ----------
        The setup argument that is a list is separated from the single value
        filters which are put into the kwargs.

        limit : int
        offset : int
        setup : list(int), optional
        flow : int, optional
        tag : str, optional

        Returns
        -------
        list
            setups that match the filters, going from id to the OpenMLSetup object.
        """
        api_call = self._build_url(limit, offset, setup=setup, flow=flow, tag=tag)
        setup_response = self._http.get(api_call)
        xml_content = setup_response.text

        return self._parse_list_xml(xml_content)

    def _build_url(
        self,
        limit: int,
        offset: int,
        *,
        setup: Iterable[int] | None = None,
        flow: int | None = None,
        tag: str | None = None,
    ) -> str:
        """Construct an OpenML Setup API URL with filtering parameters.

        Parameters
        ----------
        The setup argument that is a list is separated from the single value
        filters which are put into the kwargs.

        limit : int
        offset : int
        setup : list(int), optional
        flow : int, optional
        tag : str, optional

        Returns
        -------
        str
            A relative API path suitable for an OpenML HTTP request.
        """
        api_call = "setup/list"
        if limit is not None:
            api_call += f"/limit/{limit}"
        if offset is not None:
            api_call += f"/offset/{offset}"
        if setup is not None:
            api_call += f"/setup/{','.join([str(int(i)) for i in setup])}"
        if flow is not None:
            api_call += f"/flow/{flow}"
        if tag is not None:
            api_call += f"/tag/{tag}"

        return api_call

    def _parse_list_xml(self, xml_content: str) -> builtins.list[OpenMLSetup]:
        """Helper function to parse API calls which are lists of setups"""
        setups_dict = xmltodict.parse(xml_content, force_list=("oml:setup",))
        openml_uri = "http://openml.org/openml"
        # Minimalistic check if the XML is useful
        if "oml:setups" not in setups_dict:
            raise ValueError(
                f'Error in return XML, does not contain "oml:setups": {setups_dict!s}',
            )

        if "@xmlns:oml" not in setups_dict["oml:setups"]:
            raise ValueError(
                f'Error in return XML, does not contain "oml:setups"/@xmlns:oml: {setups_dict!s}',
            )

        if setups_dict["oml:setups"]["@xmlns:oml"] != openml_uri:
            raise ValueError(
                "Error in return XML, value of  "
                '"oml:seyups"/@xmlns:oml is not '
                f'"{openml_uri}": {setups_dict!s}',
            )

        assert isinstance(setups_dict["oml:setups"]["oml:setup"], list), type(
            setups_dict["oml:setups"]
        )

        return [
            self._create_setup({"oml:setup_parameters": setup_})
            for setup_ in setups_dict["oml:setups"]["oml:setup"]
        ]

    def _create_setup(self, result_dict: dict) -> OpenMLSetup:
        """Turns an API xml result into a OpenMLSetup object (or dict)"""
        setup_id = int(result_dict["oml:setup_parameters"]["oml:setup_id"])
        flow_id = int(result_dict["oml:setup_parameters"]["oml:flow_id"])

        if "oml:parameter" not in result_dict["oml:setup_parameters"]:
            return OpenMLSetup(setup_id, flow_id, parameters=None)

        xml_parameters = result_dict["oml:setup_parameters"]["oml:parameter"]
        if isinstance(xml_parameters, dict):
            xml_parameters = [xml_parameters]
        if not isinstance(xml_parameters, list):
            raise ValueError(
                f"Expected None, list or dict, received something else: {type(xml_parameters)!s}",
            )
            
        parameters = {
            int(xml_parameter["oml:id"]): self._create_setup_parameter_from_xml(xml_parameter)
            for xml_parameter in xml_parameters
        }
        return OpenMLSetup(setup_id, flow_id, parameters)

    def _create_setup_parameter_from_xml(self, result_dict: dict[str, str]) -> OpenMLParameter:
        """Create an OpenMLParameter object or a dictionary from an API xml result."""
        return OpenMLParameter(
            input_id=int(result_dict["oml:id"]),
            flow_id=int(result_dict["oml:flow_id"]),
            flow_name=result_dict["oml:flow_name"],
            full_name=result_dict["oml:full_name"],
            parameter_name=result_dict["oml:parameter_name"],
            data_type=result_dict["oml:data_type"],
            default_value=result_dict["oml:default_value"],
            value=result_dict["oml:value"],
        )

    def get(self, setup_id: int) -> OpenMLSetup:
        """
        Downloads the setup (configuration) description from OpenML
        and returns a structured object

        Parameters
        ----------
        setup_id : int
            The Openml setup_id

        Returns
        -------
        OpenMLSetup
            An initialized OpenMLSetup object parsed from the XML
        """
        url_suffix = f"setup/{setup_id}"
        setup_response = self._http.get(url_suffix, enable_cache=True)
        xml_content = setup_response.text
        result_dict = xmltodict.parse(xml_content)

        return self._create_setup(result_dict)

    def exists(
        self,
        flow: OpenMLFlow,
        param_settings: builtins.list[dict[str, Any]],
    ) -> int | bool:
        """
        Checks whether a hyperparameter configuration already exists on the server.

        Parameters
        ----------
        flow : OpenMLFlow
            The openml flow object. Should have flow id present for the main flow
            and all subflows (i.e., it should be downloaded from the server by
            means of flow.get, and not instantiated locally)

        list :
            A list of dicts, where each dict has the following entries:
                oml:name : str: The OpenML parameter name
                oml:value : mixed: A representation of the parameter value
                oml:component : int: flow id to which the parameter belongs

        Returns
        -------
        setup_id : int
            setup id iff exists, False otherwise
        """
        if flow.flow_id is None:
            raise ValueError("Flow must have a flow_id")
        description = xmltodict.unparse(self._to_dict(flow.flow_id, param_settings), pretty=True)
        file_elements = {
            "description": ("description.arff", description),
        }

        api_call = "setup/exists/"
        setup_response = self._http.post(api_call, files=file_elements)
        xml_content = setup_response.text
        result_dict = xmltodict.parse(xml_content)

        setup_id = int(result_dict["oml:setup_exists"]["oml:id"])
        return setup_id if setup_id > 0 else False

    def _to_dict(
        self, flow_id: int, openml_parameter_settings: builtins.list[dict[str, Any]]
    ) -> OrderedDict:
        """Convert a flow ID and a list of OpenML parameter settings to
        a dictionary representation that can be serialized to XML.

        Parameters
        ----------
        flow_id : int
            ID of the flow.
        openml_parameter_settings : list[dict[str, Any]]
            A list of OpenML parameter settings.

        Returns
        -------
        OrderedDict
            A dictionary representation of the flow ID and parameter settings.
        """
        # for convenience, this function (ab)uses the run object.
        xml: OrderedDict = OrderedDict()
        xml["oml:run"] = OrderedDict()
        xml["oml:run"]["@xmlns:oml"] = "http://openml.org/openml"
        xml["oml:run"]["oml:flow_id"] = flow_id
        xml["oml:run"]["oml:parameter_setting"] = openml_parameter_settings

        return xml


class SetupV2API(ResourceV2API, SetupAPI):
    """V2 JSoN API implementation for setups."""

    def list(
        self,
        limit: int,  # noqa: ARG002
        offset: int,  # noqa: ARG002
        *,
        setup: Iterable[int] | None = None,  # noqa: ARG002
        flow: int | None = None,  # noqa: ARG002
        tag: str | None = None,  # noqa: ARG002
    ) -> builtins.list[OpenMLSetup]:
        self._not_supported(method="list")

    def get(self, setup_id: int) -> OpenMLSetup:  # noqa: ARG002
        self._not_supported(method="get")

    def exists(
        self,
        flow: OpenMLFlow,  # noqa: ARG002
        param_settings: builtins.list[dict[str, Any]],  # noqa: ARG002
    ) -> int | bool:
        self._not_supported(method="exists")
