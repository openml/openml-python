from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NoReturn

from openml.exceptions import (
    OpenMLNotAuthorizedError,
    OpenMLNotSupportedError,
    OpenMLServerError,
    OpenMLServerException,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from openml._api.clients import HTTPClient
    from openml.enums import APIVersion, ResourceType


class ResourceAPI(ABC):
    """
    Abstract base class for OpenML resource APIs.

    This class defines the common interface for interacting with OpenML
    resources (e.g., datasets, flows, runs) across different API versions.
    Concrete subclasses must implement the resource-specific operations
    such as publishing, deleting, and tagging.

    Parameters
    ----------
    http : HTTPClient
        Configured HTTP client used for communication with the OpenML API.
    minio : MinIOClient
        Configured MinIO client used for object storage operations.

    Attributes
    ----------
    api_version : APIVersion
        API version implemented by the resource.
    resource_type : ResourceType
        Type of OpenML resource handled by the implementation.
    _http : HTTPClient
        Internal HTTP client instance.
    _minio : MinIOClient or None
        Internal MinIO client instance, if provided.
    """

    api_version: APIVersion
    resource_type: ResourceType

    def __init__(self, http: HTTPClient):
        self._http = http

    @abstractmethod
    def delete(self, resource_id: int) -> bool:
        """
        Delete a resource by its identifier.

        Parameters
        ----------
        resource_id : int
            Unique identifier of the resource to delete.

        Returns
        -------
        bool
            ``True`` if the deletion was successful.

        Notes
        -----
        Concrete subclasses must implement this method.
        """

    @abstractmethod
    def publish(self, path: str, files: Mapping[str, Any] | None) -> int:
        """
        Publish a new resource to the OpenML server.

        Parameters
        ----------
        path : str
            API endpoint path used for publishing the resource.
        files : Mapping of str to Any or None
            Files or payload data required for publishing. The structure
            depends on the resource type.

        Returns
        -------
        int
            Identifier of the newly created resource.

        Notes
        -----
        Concrete subclasses must implement this method.
        """

    @abstractmethod
    def tag(self, resource_id: int, tag: str) -> list[str]:
        """
        Add a tag to a resource.

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

        Notes
        -----
        Concrete subclasses must implement this method.
        """

    @abstractmethod
    def untag(self, resource_id: int, tag: str) -> list[str]:
        """
        Remove a tag from a resource.

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

        Notes
        -----
        Concrete subclasses must implement this method.
        """

    @abstractmethod
    def _get_endpoint_name(self) -> str:
        """
        Return the endpoint name for the current resource type.

        Returns
        -------
        str
            Endpoint segment used in API paths.

        Notes
        -----
        Datasets use the special endpoint name ``"data"`` instead of
        their enum value.
        """

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

    def _not_supported(self, *, method: str) -> NoReturn:
        """
        Raise an error indicating that a method is not supported.

        Parameters
        ----------
        method : str
            Name of the unsupported method.

        Raises
        ------
        OpenMLNotSupportedError
            If the current API version does not support the requested method
            for the given resource type.
        """
        version = getattr(self.api_version, "value", "unknown")
        resource = getattr(self.resource_type, "value", "unknown")

        raise OpenMLNotSupportedError(
            f"{self.__class__.__name__}: "
            f"{version} API does not support `{method}` "
            f"for resource `{resource}`"
        )
