from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from requests import Response

    from openml._api.http import HTTPClient
    from openml.datasets.dataset import OpenMLDataset
    from openml.tasks.task import OpenMLTask


class ResourceAPI:
    def __init__(self, http: HTTPClient):
        self._http = http


class DatasetsAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(self, dataset_id: int) -> OpenMLDataset | tuple[OpenMLDataset, Response]: ...


class TasksAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(
        self,
        task_id: int,
        *,
        return_response: bool = False,
    ) -> OpenMLTask | tuple[OpenMLTask, Response]: ...


class StudiesAPI(ResourceAPI, ABC):
    @abstractmethod
    def list(  # noqa: PLR0913
        self,
        limit: int | None = None,
        offset: int | None = None,
        status: str | None = None,
        main_entity_type: str | None = None,
        uploader: list[int] | None = None,
        benchmark_suite: int | None = None,
    ) -> Any: ...
