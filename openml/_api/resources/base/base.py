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
    DATASET = "dataset"
    TASK = "task"
    TASK_TYPE = "task_type"
    EVALUATION_MEASURE = "evaluation_measure"
    ESTIMATION_PROCEDURE = "estimation_procedure"
    EVALUATION = "evaluation"
    FLOW = "flow"
    STUDY = "study"
    RUN = "run"
    SETUP = "setup"
    USER = "user"


class ResourceAPI(ABC):
    api_version: APIVersion
    resource_type: ResourceType

    def __init__(self, http: HTTPClient):
        self._http = http

    def _get_not_implemented_message(self, method_name: str | None = None) -> str:
        version = getattr(self.api_version, "name", "Unknown version")
        resource = getattr(self.resource_type, "name", "Unknown resource")
        method_info = f" Method: {method_name}" if method_name else ""
        return (
            f"{self.__class__.__name__}: {version} API does not support this "
            f"functionality for resource: {resource}.{method_info}"
        )

    @abstractmethod
    def delete(self, resource_id: int) -> bool: ...

    @abstractmethod
    def publish(self) -> None: ...
