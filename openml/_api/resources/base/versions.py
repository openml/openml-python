from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from xml.parsers.expat import ExpatError

import xmltodict

from openml.enums import APIVersion, ResourceType
from openml.exceptions import (
    OpenMLServerException,
)

from .base import ResourceAPI

_LEGAL_RESOURCES_DELETE = [
    ResourceType.DATASET,
    ResourceType.TASK,
    ResourceType.FLOW,
    ResourceType.STUDY,
    ResourceType.RUN,
    ResourceType.USER,
]

_LEGAL_RESOURCES_TAG = [
    ResourceType.DATASET,
    ResourceType.TASK,
    ResourceType.FLOW,
    ResourceType.SETUP,
    ResourceType.RUN,
]


class ResourceV1API(ResourceAPI):
    """
    Version 1 implementation of the OpenML resource API.

    This class provides XML-based implementations for publishing,
    deleting, tagging, and untagging resources using the V1 API
    endpoints. Responses are parsed using ``xmltodict``.

    Notes
    -----
    V1 endpoints expect and return XML. Error handling follows the
    legacy OpenML server behavior and maps specific error codes to
    more descriptive exceptions where appropriate.
    """

    api_version: APIVersion = APIVersion.V1

    def publish(self, path: str, files: Mapping[str, Any] | None) -> int:
        """
        Publish a new resource using the V1 API.

        Parameters
        ----------
        path : str
            API endpoint path for the upload.
        files : Mapping of str to Any or None
            Files to upload as part of the request payload.

        Returns
        -------
        int
            Identifier of the newly created resource.

        Raises
        ------
        ValueError
            If the server response does not contain a valid resource ID.
        OpenMLServerException
            If the server returns an error during upload.
        """
        response = self._http.post(path, files=files)
        parsed_response = self._parse_xml_response(response.content)
        return self._extract_id_from_upload(parsed_response)

    def delete(self, resource_id: int) -> bool:
        """
        Delete a resource using the V1 API.

        Parameters
        ----------
        resource_id : int
            Identifier of the resource to delete.

        Returns
        -------
        bool
            ``True`` if the server confirms successful deletion.

        Raises
        ------
        ValueError
            If the resource type is not supported for deletion.
        OpenMLNotAuthorizedError
            If the user is not permitted to delete the resource.
        OpenMLServerError
            If deletion fails for an unknown reason.
        OpenMLServerException
            For other server-side errors.
        """
        if self.resource_type not in _LEGAL_RESOURCES_DELETE:
            raise ValueError(f"Can't delete a {self.resource_type.value}")

        endpoint_name = self._get_endpoint_name()
        path = f"{endpoint_name}/{resource_id}"
        try:
            response = self._http.delete(path)
            result = self._parse_xml_response(response.content)
            return f"oml:{endpoint_name}_delete" in result
        except OpenMLServerException as e:
            self._handle_delete_exception(endpoint_name, e)
            raise

    def tag(self, resource_id: int, tag: str) -> list[str]:
        """
        Add a tag to a resource using the V1 API.

        Parameters
        ----------
        resource_id : int
            Identifier of the resource to tag.
        tag : str
            Tag to associate with the resource.

        Returns
        -------
        list of str
            Updated list of tags assigned to the resource.

        Raises
        ------
        ValueError
            If the resource type does not support tagging.
        OpenMLServerException
            If the server returns an error.
        """
        if self.resource_type not in _LEGAL_RESOURCES_TAG:
            raise ValueError(f"Can't tag a {self.resource_type.value}")

        endpoint_name = self._get_endpoint_name()
        path = f"{endpoint_name}/tag"
        data = {f"{endpoint_name}_id": resource_id, "tag": tag}
        response = self._http.post(path, data=data)

        parsed_response = self._parse_xml_response(response.content, force_list={"oml:tag"})
        result = parsed_response[f"oml:{endpoint_name}_tag"]
        tags: list[str] = result.get("oml:tag", [])

        return tags

    def untag(self, resource_id: int, tag: str) -> list[str]:
        """
        Remove a tag from a resource using the V1 API.

        Parameters
        ----------
        resource_id : int
            Identifier of the resource to untag.
        tag : str
            Tag to remove from the resource.

        Returns
        -------
        list of str
            Updated list of tags assigned to the resource.

        Raises
        ------
        ValueError
            If the resource type does not support tagging.
        OpenMLServerException
            If the server returns an error.
        """
        if self.resource_type not in _LEGAL_RESOURCES_TAG:
            raise ValueError(f"Can't untag a {self.resource_type.value}")

        endpoint_name = self._get_endpoint_name()
        path = f"{endpoint_name}/untag"
        data = {f"{endpoint_name}_id": resource_id, "tag": tag}
        response = self._http.post(path, data=data)

        parsed_response = self._parse_xml_response(response.content, force_list={"oml:tag"})
        result = parsed_response[f"oml:{endpoint_name}_untag"]
        tags: list[str] = result.get("oml:tag", [])

        return tags

    def _parse_xml_response(self, payload: bytes | str, **kwargs: Any) -> Mapping[str, Any]:
        try:
            parsed_response: Mapping[str, Any] = xmltodict.parse(payload, **kwargs)
            return parsed_response
        except ExpatError:
            payload_text = (
                payload.decode("utf-8", errors="ignore") if isinstance(payload, bytes) else payload
            )
            xml_start = payload_text.find("<?xml")
            if xml_start == -1:
                xml_start = payload_text.find("<oml:")
            if xml_start == -1:
                raise

            xml_text = payload_text[xml_start:]
            parsed_fallback: Mapping[str, Any] = xmltodict.parse(xml_text, **kwargs)
            return parsed_fallback

    def _get_endpoint_name(self) -> str:
        if self.resource_type == ResourceType.DATASET:
            return "data"
        endpoint_name = self.resource_type.value
        if not isinstance(endpoint_name, str):
            raise TypeError(f"Unexpected endpoint type: {type(endpoint_name)}")
        return endpoint_name

    def _extract_id_from_upload(self, parsed: Mapping[str, Any]) -> int:
        """
        Extract the resource identifier from an XML upload response.

        Parameters
        ----------
        parsed : Mapping of str to Any
            Parsed XML response as returned by ``xmltodict.parse``.

        Returns
        -------
        int
            Extracted resource identifier.

        Raises
        ------
        ValueError
            If the response structure is unexpected or no identifier
            can be found.
        """
        # reads id from upload response
        # actual parsed dict: {"oml:upload_flow": {"@xmlns:oml": "...", "oml:id": "42"}}

        # xmltodict always gives exactly one root key
        ((_, root_value),) = parsed.items()

        if not isinstance(root_value, Mapping):
            raise ValueError("Unexpected XML structure")

        # 1. Specifically look for keys ending in _id or id (e.g., oml:id, oml:run_id)
        for k, v in root_value.items():
            if (
                (k.endswith(("id", "_id")) or "id" in k.lower())
                and isinstance(v, (str, int))
                and str(v).isdigit()
            ):
                return int(v)

        # 2. Fallback: check all values for numeric/string IDs, excluding xmlns or URLs
        for v in root_value.values():
            if isinstance(v, (str, int)):
                val_str = str(v)
                if val_str.isdigit():
                    return int(val_str)

        raise ValueError(f"No ID found in upload response: {root_value}")


class ResourceV2API(ResourceAPI):
    """
    Version 2 implementation of the OpenML resource API.

    This class represents the V2 API for resources. Operations such as
    publishing, deleting, tagging, and untagging are currently not
    supported and will raise ``OpenMLNotSupportedError``.
    """

    api_version: APIVersion = APIVersion.V2

    def publish(self, path: str, files: Mapping[str, Any] | None) -> int:  # noqa: ARG002
        self._not_supported(method="publish")

    def delete(self, resource_id: int) -> bool:  # noqa: ARG002
        self._not_supported(method="delete")

    def tag(self, resource_id: int, tag: str) -> list[str]:  # noqa: ARG002
        self._not_supported(method="tag")

    def untag(self, resource_id: int, tag: str) -> list[str]:  # noqa: ARG002
        self._not_supported(method="untag")

    def _get_endpoint_name(self) -> str:
        endpoint_name = self.resource_type.value
        if not isinstance(endpoint_name, str):
            raise TypeError(f"Unexpected endpoint type: {type(endpoint_name)}")
        return endpoint_name
