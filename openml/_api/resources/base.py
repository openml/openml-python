from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response

    from openml._api.http import HTTPClient
    from openml.datasets.dataset import OpenMLDataset
    from openml.runs.run import OpenMLRun
    from openml.tasks.task import OpenMLTask, TaskType


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


class RunsAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(self, run_id: int) -> OpenMLRun: ...

    @abstractmethod
    def list(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        ids: list | None = None,
        task: list | None = None,
        setup: list | None = None,
        flow: list | None = None,
        uploader: list | None = None,
        study: int | None = None,
        tag: str | None = None,
        display_errors: bool = False,
        task_type: TaskType | int | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def delete(self, run_id: int) -> bool: ...

    @abstractmethod
    def create(self, run: OpenMLRun) -> OpenMLRun: ...
