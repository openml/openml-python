from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import xmltodict

from openml._api.resources.base import FlowsAPI
from openml.exceptions import OpenMLServerException
from openml.flows.flow import OpenMLFlow


class FlowsV1(FlowsAPI):
    def get(
        self,
        flow_id: int,
    ) -> OpenMLFlow:
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
        return OpenMLFlow._from_dict(xmltodict.parse(flow_xml))

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

        data = {"name": name, "external_version": external_version, "api_key": self._http.key}
        # Avoid duplicating base_url when server already contains the API path
        server = self._http.server
        base = self._http.base_url
        if base and base.strip("/") in server:
            url = server.rstrip("/") + "/flow/exists"
            response = requests.post(
                url, data=data, headers=self._http.headers, timeout=self._http.timeout
            )
            xml_response = response.text
        else:
            xml_response = self._http.post("flow/exists", data=data).text
        result_dict = xmltodict.parse(xml_response)
        # Detect error payloads and raise
        if "oml:error" in result_dict:
            err = result_dict["oml:error"]
            code = int(err.get("oml:code", 0)) if "oml:code" in err else None
            message = err.get("oml:message", "Server returned an error")
            raise OpenMLServerException(message=message, code=code)

        flow_id = int(result_dict["oml:flow_exists"]["oml:id"])
        return flow_id if flow_id > 0 else False

    def list(
        self,
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

        server = self._http.server
        base = self._http.base_url
        if base and base.strip("/") in server:
            url = server.rstrip("/") + "/" + api_call
            response = requests.get(
                url,
                headers=self._http.headers,
                params={"api_key": self._http.key},
                timeout=self._http.timeout,
            )
            xml_string = response.text
        else:
            response = self._http.get(api_call, use_api_key=True)
            xml_string = response.text
        flows_dict = xmltodict.parse(xml_string, force_list=("oml:flow",))

        if "oml:error" in flows_dict:
            err = flows_dict["oml:error"]
            code = int(err.get("oml:code", 0)) if "oml:code" in err else None
            message = err.get("oml:message", "Server returned an error")
            raise OpenMLServerException(message=message, code=code)

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

        # POST to server (multipart/files). Ensure api_key is sent in the form data.
        files = file_elements
        data = {"api_key": self._http.key}
        # If server already contains base path, post directly with requests to avoid double base_url
        server = self._http.server
        base = self._http.base_url
        if base and base.strip("/") in server:
            url = server.rstrip("/") + "/flow"
            response = requests.post(
                url, files=files, data=data, headers=self._http.headers, timeout=self._http.timeout
            )
        else:
            response = self._http.post("flow", files=files, data=data)

        parsed = xmltodict.parse(response.text)
        if "oml:error" in parsed:
            err = parsed["oml:error"]
            code = int(err.get("oml:code", 0)) if "oml:code" in err else None
            message = err.get("oml:message", "Server returned an error")
            raise OpenMLServerException(message=message, code=code)

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
    ) -> OpenMLFlow:
        """Get a flow from the OpenML v2 server.

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
        response = self._http.get(f"flows/{flow_id}/")
        flow_json = response.json()

        # Convert v2 JSON to v1-compatible dict for OpenMLFlow._from_dict()
        flow_dict = self._convert_v2_to_v1_format(flow_json)
        return OpenMLFlow._from_dict(flow_dict)

    def exists(self, name: str, external_version: str) -> int | bool:
        """Check if a flow exists on the OpenML v2 server.

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

        try:
            response = self._http.get(f"flows/exists/{name}/{external_version}/")
            result = response.json()
            flow_id: int | bool = result.get("flow_id", False)
            return flow_id
        except (requests.exceptions.HTTPError, KeyError):
            # v2 returns 404 when flow doesn't exist
            return False

    def list(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        uploader: str | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError("flows (list) not yet implemented in v2 server")

    def create(self, flow: OpenMLFlow) -> OpenMLFlow:
        raise NotImplementedError("POST /flows (create) not yet implemented in v2 server")

    def delete(self, flow_id: int) -> None:
        raise NotImplementedError("DELETE /flows/{id} not yet implemented in v2 server")

    @staticmethod
    def _convert_v2_to_v1_format(v2_json: dict[str, Any]) -> dict[str, dict]:
        """Convert v2 JSON response to v1 XML-dict format for OpenMLFlow._from_dict().

        Parameters
        ----------
        v2_json : dict
            The v2 JSON response from the server.

        Returns
        -------
        dict
            A dictionary matching the v1 XML structure expected by OpenMLFlow._from_dict().
        """
        # Map v2 JSON fields to v1 XML structure with oml: namespace
        flow_dict = {
            "oml:flow": {
                "@xmlns:oml": "http://openml.org/openml",
                "oml:id": str(v2_json.get("id", "0")),
                "oml:uploader": str(v2_json.get("uploader", "")),
                "oml:name": v2_json.get("name", ""),
                "oml:version": str(v2_json.get("version", "")),
                "oml:external_version": v2_json.get("external_version", ""),
                "oml:description": v2_json.get("description", ""),
                "oml:upload_date": (
                    v2_json.get("upload_date", "").replace("T", " ")
                    if v2_json.get("upload_date")
                    else ""
                ),
                "oml:language": v2_json.get("language", ""),
                "oml:dependencies": v2_json.get("dependencies", ""),
            }
        }

        # Add optional fields
        if "class_name" in v2_json:
            flow_dict["oml:flow"]["oml:class_name"] = v2_json["class_name"]
        if "custom_name" in v2_json:
            flow_dict["oml:flow"]["oml:custom_name"] = v2_json["custom_name"]

        # Convert parameters from v2 array to v1 format
        if v2_json.get("parameter"):
            flow_dict["oml:flow"]["oml:parameter"] = [
                {
                    "oml:name": param.get("name", ""),
                    "oml:data_type": param.get("data_type", ""),
                    "oml:default_value": str(param.get("default_value", "")),
                    "oml:description": param.get("description", ""),
                }
                for param in v2_json["parameter"]
            ]

        # Convert subflows from v2 to v1 components format
        if v2_json.get("subflows"):
            flow_dict["oml:flow"]["oml:component"] = [
                {
                    "oml:identifier": subflow.get("identifier", ""),
                    "oml:flow": FlowsV2._convert_v2_to_v1_format(subflow["flow"])["oml:flow"],
                }
                for subflow in v2_json["subflows"]
            ]

        # Convert tags from v2 array to v1 format
        if v2_json.get("tag"):
            flow_dict["oml:flow"]["oml:tag"] = v2_json["tag"]

        return flow_dict
