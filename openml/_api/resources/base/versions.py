from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import xmltodict

from openml._api.resources.base import ResourceAPI
from openml.enums import APIVersion, ResourceType
from openml.exceptions import (
    OpenMLServerException,
)

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
    ResourceType.STUDY,
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
        parsed_response = xmltodict.parse(response.content)
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
            result = xmltodict.parse(response.content)
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

        parsed_response = xmltodict.parse(response.content, force_list={"oml:tag"})
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

        parsed_response = xmltodict.parse(response.content, force_list={"oml:tag"})
        result = parsed_response[f"oml:{endpoint_name}_untag"]
        tags: list[str] = result.get("oml:tag", [])

        return tags

    def _get_endpoint_name(self) -> str:
        if self.resource_type == ResourceType.DATASET:
            return "data"
        return cast("str", self.resource_type.value)

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

        # Look for oml:id directly in the root value
        if "oml:id" in root_value:
            id_value = root_value["oml:id"]
            if isinstance(id_value, (str, int)):
                return int(id_value)

        # Fallback: check all values for numeric/string IDs
        for v in root_value.values():
            if isinstance(v, (str, int)):
                return int(v)

        raise ValueError("No ID found in upload response")


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
        return cast("str", self.resource_type.value)
