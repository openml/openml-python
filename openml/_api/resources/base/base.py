from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openml._api.clients import HTTPClient


class APIVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


class ResourceType(str, Enum):
    DATASETS = "datasets"
    TASKS = "tasks"


class ResourceAPI(ABC):
    api_version: APIVersion | None = None
    resource_type: ResourceType | None = None

    def __init__(self, http: HTTPClient):
        self._http = http

    def _raise_not_implemented_error(self, method_name: str | None = None) -> None:
        version = getattr(self.api_version, "name", "Unknown version")
        resource = getattr(self.resource_type, "name", "Unknown resource")
        method_info = f" Method: {method_name}" if method_name else ""
        raise NotImplementedError(
            f"{self.__class__.__name__}: {version} API does not support this "
            f"functionality for resource: {resource}.{method_info}"
        )

    @abstractmethod
    def delete(self) -> None: ...

    @abstractmethod
    def publish(self) -> None: ...
