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
        response = self._http.get(f"flow/{flow_id}")
        flow_xml = response.text
        flow = OpenMLFlow._from_dict(xmltodict.parse(flow_xml))
        if return_response:
            return flow, response
        return flow

    def exists(self, name: str, external_version: str) -> int | bool:
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
