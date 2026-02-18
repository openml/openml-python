from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NoReturn

from openml.exceptions import OpenMLNotSupportedError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from openml._api.clients import HTTPClient, MinIOClient
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

    def __init__(self, http: HTTPClient, minio: MinIOClient):
        self._http = http
        self._minio = minio

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
