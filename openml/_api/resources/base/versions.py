from __future__ import annotations

import xmltodict

from openml._api.resources.base import APIVersion, ResourceAPI, ResourceType
from openml.exceptions import (
    OpenMLNotAuthorizedError,
    OpenMLServerError,
    OpenMLServerException,
)


class ResourceV1(ResourceAPI):
    api_version: APIVersion = APIVersion.V1

    def delete(self, resource_id: int) -> bool:
        if self.resource_type == ResourceType.DATASET:
            resource_type = "data"
        else:
            resource_type = self.resource_type.name

        legal_resources = {
            "data",
            "flow",
            "task",
            "run",
            "study",
            "user",
        }
        if resource_type not in legal_resources:
            raise ValueError(f"Can't delete a {resource_type}")

        url_suffix = f"{resource_type}/{resource_id}"
        try:
            response = self._http.delete(url_suffix)
            result = xmltodict.parse(response.content)
            return f"oml:{resource_type}_delete" in result
        except OpenMLServerException as e:
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
            if e.code in by_other_user:
                raise OpenMLNotAuthorizedError(
                    message=(
                        f"The {resource_type} can not be deleted "
                        "because it was not uploaded by you."
                    ),
                ) from e
            if e.code in has_dependent_entities:
                raise OpenMLNotAuthorizedError(
                    message=(
                        f"The {resource_type} can not be deleted because "
                        f"it still has associated entities: {e.message}"
                    ),
                ) from e
            if e.code in unknown_reason:
                raise OpenMLServerError(
                    message=(
                        f"The {resource_type} can not be deleted for unknown reason,"
                        " please open an issue at: https://github.com/openml/openml/issues/new"
                    ),
                ) from e
            raise e

    def publish(self) -> None:
        pass


class ResourceV2(ResourceAPI):
    api_version: APIVersion = APIVersion.V2

    def delete(self, resource_id: int) -> bool:
        raise NotImplementedError(self._get_not_implemented_message("publish"))

    def publish(self) -> None:
        raise NotImplementedError(self._get_not_implemented_message("publish"))
