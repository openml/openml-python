from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, NoReturn

from openml.exceptions import OpenMLNotSupportedError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

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

    @abstractmethod
    def delete(self, resource_id: int) -> bool: ...

    @abstractmethod
    def publish(self, path: str, files: Mapping[str, Any] | None) -> int: ...

    @abstractmethod
    def tag(self, resource_id: int, tag: str) -> list[str]: ...

    @abstractmethod
    def untag(self, resource_id: int, tag: str) -> list[str]: ...

    def _not_supported(self, *, method: str) -> NoReturn:
        version = getattr(self.api_version, "value", "unknown")
        resource = getattr(self.resource_type, "value", "unknown")

        raise OpenMLNotSupportedError(
            f"{self.__class__.__name__}: "
            f"{version} API does not support `{method}` "
            f"for resource `{resource}`"
        )
