from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import xmltodict

from openml.enums import APIVersion, ResourceType
from openml.exceptions import (
    OpenMLNotAuthorizedError,
    OpenMLServerError,
    OpenMLServerException,
)

from .base import ResourceAPI


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
        resource_type = self._get_endpoint_name()

        legal_resources = {"data", "flow", "task", "run", "study", "user"}
        if resource_type not in legal_resources:
            raise ValueError(f"Can't delete a {resource_type}")

        path = f"{resource_type}/{resource_id}"
        try:
            response = self._http.delete(path)
            result = xmltodict.parse(response.content)
            return f"oml:{resource_type}_delete" in result
        except OpenMLServerException as e:
            self._handle_delete_exception(resource_type, e)
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
        resource_type = self._get_endpoint_name()

        legal_resources = {"data", "task", "flow", "setup", "run"}
        if resource_type not in legal_resources:
            raise ValueError(f"Can't tag a {resource_type}")

        path = f"{resource_type}/tag"
        data = {f"{resource_type}_id": resource_id, "tag": tag}
        response = self._http.post(path, data=data)

        main_tag = f"oml:{resource_type}_tag"
        parsed_response = xmltodict.parse(response.content, force_list={"oml:tag"})
        result = parsed_response[main_tag]
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
        resource_type = self._get_endpoint_name()

        legal_resources = {"data", "task", "flow", "setup", "run"}
        if resource_type not in legal_resources:
            raise ValueError(f"Can't untag a {resource_type}")

        path = f"{resource_type}/untag"
        data = {f"{resource_type}_id": resource_id, "tag": tag}
        response = self._http.post(path, data=data)

        main_tag = f"oml:{resource_type}_untag"
        parsed_response = xmltodict.parse(response.content, force_list={"oml:tag"})
        result = parsed_response[main_tag]
        tags: list[str] = result.get("oml:tag", [])

        return tags

    def _get_endpoint_name(self) -> str:
        """
        Return the V1 endpoint name for the current resource type.

        Returns
        -------
        str
            Endpoint segment used in V1 API paths.

        Notes
        -----
        Datasets use the special endpoint name ``"data"`` instead of
        their enum value.
        """
        if self.resource_type == ResourceType.DATASET:
            return "data"
        return cast("str", self.resource_type.value)

    def _handle_delete_exception(
        self, resource_type: str, exception: OpenMLServerException
    ) -> None:
        """
        Map V1 deletion error codes to more specific exceptions.

        Parameters
        ----------
        resource_type : str
            Endpoint name of the resource type.
        exception : OpenMLServerException
            Original exception raised during deletion.

        Raises
        ------
        OpenMLNotAuthorizedError
            If the resource cannot be deleted due to ownership or
            dependent entities.
        OpenMLServerError
            If deletion fails for an unknown reason.
        OpenMLServerException
            If the error code is not specially handled.
        """
        # https://github.com/openml/OpenML/blob/21f6188d08ac24fcd2df06ab94cf421c946971b0/openml_OS/views/pages/api_new/v1/xml/pre.php
        # Most exceptions are descriptive enough to be raised as their standard
        # OpenMLServerException, however there are two cases where we add information:
        #  - a generic "failed" message, we direct them to the right issue board
        #  - when the user successfully authenticates with the server,
        #    but user is not allowed to take the requested action,
        #    in which case we specify a OpenMLNotAuthorizedError.
        by_other_user = [323, 353, 393, 453, 594]
        has_dependent_entities = [324, 326, 327, 328, 354, 454, 464, 595]
        unknown_reason = [325, 355, 394, 455, 593]
        if exception.code in by_other_user:
            raise OpenMLNotAuthorizedError(
                message=(
                    f"The {resource_type} can not be deleted because it was not uploaded by you."
                ),
            ) from exception
        if exception.code in has_dependent_entities:
            raise OpenMLNotAuthorizedError(
                message=(
                    f"The {resource_type} can not be deleted because "
                    f"it still has associated entities: {exception.message}"
                ),
            ) from exception
        if exception.code in unknown_reason:
            raise OpenMLServerError(
                message=(
                    f"The {resource_type} can not be deleted for unknown reason,"
                    " please open an issue at: https://github.com/openml/openml/issues/new"
                ),
            ) from exception
        raise exception

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
