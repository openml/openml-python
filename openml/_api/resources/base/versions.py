from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xmltodict

from openml._api.resources.base import APIVersion, ResourceAPI, ResourceType
from openml.exceptions import (
    OpenMLNotAuthorizedError,
    OpenMLServerError,
    OpenMLServerException,
)


class ResourceV1(ResourceAPI):
    api_version: APIVersion = APIVersion.V1

    def publish(self, path: str, files: Mapping[str, Any] | None) -> int:
        response = self._http.post(path, files=files)
        parsed_response = xmltodict.parse(response.content)
        return self._extract_id_from_upload(parsed_response)

    def delete(self, resource_id: int) -> bool:
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
        resource_type = self._get_endpoint_name()

        legal_resources = {"data", "task", "flow", "setup", "run"}
        if resource_type not in legal_resources:
            raise ValueError(f"Can't tag a {resource_type}")

        path = f"{resource_type}/untag"
        data = {f"{resource_type}_id": resource_id, "tag": tag}
        response = self._http.post(path, data=data)

        main_tag = f"oml:{resource_type}_untag"
        parsed_response = xmltodict.parse(response.content, force_list={"oml:tag"})
        result = parsed_response[main_tag]
        tags: list[str] = result.get("oml:tag", [])

        return tags

    def _get_endpoint_name(self) -> str:
        if self.resource_type == ResourceType.DATASET:
            return "data"
        return self.resource_type.name

    def _handle_delete_exception(
        self, resource_type: str, exception: OpenMLServerException
    ) -> None:
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
        # reads id from
        # sample parsed dict: {"oml:openml": {"oml:upload_flow": {"oml:id": "42"}}}

        # xmltodict always gives exactly one root key
        ((_, root_value),) = parsed.items()

        if not isinstance(root_value, Mapping):
            raise ValueError("Unexpected XML structure")

        # upload node (e.g. oml:upload_task, oml:study_upload, ...)
        ((_, upload_value),) = root_value.items()

        if not isinstance(upload_value, Mapping):
            raise ValueError("Unexpected upload node structure")

        # ID is the only leaf value
        for v in upload_value.values():
            if isinstance(v, (str, int)):
                return int(v)

        raise ValueError("No ID found in upload response")


class ResourceV2(ResourceAPI):
    api_version: APIVersion = APIVersion.V2

    def publish(self, path: str, files: Mapping[str, Any] | None) -> int:
        raise NotImplementedError(self._get_not_implemented_message("publish"))

    def delete(self, resource_id: int) -> bool:
        raise NotImplementedError(self._get_not_implemented_message("delete"))

    def tag(self, resource_id: int, tag: str) -> list[str]:
        raise NotImplementedError(self._get_not_implemented_message("untag"))

    def untag(self, resource_id: int, tag: str) -> list[str]:
        raise NotImplementedError(self._get_not_implemented_message("untag"))
