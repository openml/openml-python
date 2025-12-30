from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openml._api.http import HTTPClient


class ResourceAPI:
    def __init__(self, http: HTTPClient):
        self._http = http


class DatasetsAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(self, id: int) -> dict: ...


class TasksAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(self, id: int) -> dict: ...
