from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xmltodict

from openml._api.resources.base import FlowsAPI
from openml.flows.flow import OpenMLFlow

if TYPE_CHECKING:
    from requests import Response


class FlowsV1(FlowsAPI):
    def get(
        self,
        flow_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLFlow | tuple[OpenMLFlow, Response]:
        """Get a flow from the OpenML server.

        Parameters
        ----------
        flow_id : int
            The ID of the flow to retrieve.
        return_response : bool, optional (default=False)
            Whether to return the raw response object along with the flow.

        Returns
        -------
        OpenMLFlow | tuple[OpenMLFlow, Response]
            The retrieved flow object, and optionally the raw response.
        """
        response = self._http.get(f"flow/{flow_id}")
        flow_xml = response.text
        flow = OpenMLFlow._from_dict(xmltodict.parse(flow_xml))
        if return_response:
            return flow, response
        return flow

    def exists(self, name: str, external_version: str) -> int | bool:
        """Check if a flow exists on the OpenML server.

        Parameters
        ----------
        name : str
            The name of the flow.
        external_version : str
            The external version of the flow.

        Returns
        -------
        int | bool
            The flow ID if the flow exists, False otherwise.
        """
        if not (isinstance(name, str) and len(name) > 0):
            raise ValueError("Argument 'name' should be a non-empty string")
        if not (isinstance(external_version, str) and len(external_version) > 0):
            raise ValueError("Argument 'version' should be a non-empty string")

        xml_response = self._http.post(
            "flow/exists", data={"name": name, "external_version": external_version}
        ).text
        result_dict = xmltodict.parse(xml_response)
        flow_id = int(result_dict["oml:flow_exists"]["oml:id"])
        return flow_id if flow_id > 0 else False

    def list_page(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        uploader: str | None = None,
    ) -> pd.DataFrame:
        """List flows on the OpenML server.

        Parameters
        ----------
        limit : int, optional
            The maximum number of flows to return.
            By default, all flows are returned.
        offset : int, optional
            The number of flows to skip before starting to collect the result set.
            By default, no flows are skipped.
        tag : str, optional
            The tag to filter flows by.
            By default, no tag filtering is applied.
        uploader : str, optional
            The user to filter flows by.
            By default, no user filtering is applied.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the list of flows.
        """
        api_call = "flow/list"
        if limit is not None:
            api_call += f"/limit/{limit}"
        if offset is not None:
            api_call += f"/offset/{offset}"
        if tag is not None:
            api_call += f"/tag/{tag}"
        if uploader is not None:
            api_call += f"/uploader/{uploader}"

        xml_string = self._http.get(api_call).text
        flows_dict = xmltodict.parse(xml_string, force_list=("oml:flow",))

        assert isinstance(flows_dict["oml:flows"]["oml:flow"], list), type(flows_dict["oml:flows"])
        assert flows_dict["oml:flows"]["@xmlns:oml"] == "http://openml.org/openml", flows_dict[
            "oml:flows"
        ]["@xmlns:oml"]

        flows: dict[int, dict[str, Any]] = {}
        for flow_ in flows_dict["oml:flows"]["oml:flow"]:
            fid = int(flow_["oml:id"])
            flow_row = {
                "id": fid,
                "full_name": flow_["oml:full_name"],
                "name": flow_["oml:name"],
                "version": flow_["oml:version"],
                "external_version": flow_["oml:external_version"],
                "uploader": flow_["oml:uploader"],
            }
            flows[fid] = flow_row

        return pd.DataFrame.from_dict(flows, orient="index")

    def create(self, flow: OpenMLFlow) -> OpenMLFlow:
        """Create a new flow on the OpenML server.

        under development , not fully functional yet

        Parameters
        ----------
        flow : OpenMLFlow
            The flow object to upload to the server.

        Returns
        -------
        OpenMLFlow
            The updated flow object with the server-assigned flow_id.
        """
        from openml.extensions import Extension

        # Check if flow is an OpenMLFlow or a compatible extension object
        if not isinstance(flow, OpenMLFlow) and not isinstance(flow, Extension):
            raise TypeError(f"Flow must be an OpenMLFlow or Extension instance, got {type(flow)}")

        # Get file elements for upload (includes XML description if not provided)
        file_elements = flow._get_file_elements()
        if "description" not in file_elements:
            file_elements["description"] = flow._to_xml()

        # POST to server
        response = self._http.post("flow", data=file_elements)

        # Parse response and update flow with server-assigned ID
        xml_response = xmltodict.parse(response.text)
        flow._parse_publish_response(xml_response)

        return flow

    def delete(self, flow_id: int) -> None:
        """Delete a flow from the OpenML server.

        Parameters
        ----------
        flow_id : int
            The ID of the flow to delete.
        """
        self._http.delete(f"flow/{flow_id}")


class FlowsV2(FlowsAPI):
    def get(
        self,
        flow_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLFlow | tuple[OpenMLFlow, Response]:
        raise NotImplementedError

    def exists(self, name: str, external_version: str) -> int | bool:
        raise NotImplementedError

    def list_page(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        uploader: str | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def create(self, flow: OpenMLFlow) -> OpenMLFlow:
        raise NotImplementedError

    def delete(self, flow_id: int) -> None:
        raise NotImplementedError
