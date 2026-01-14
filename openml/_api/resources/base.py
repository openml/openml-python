from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response

    from openml._api.http import HTTPClient
    from openml.datasets.dataset import OpenMLDataset
    from openml.flows.flow import OpenMLFlow
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


class FlowsAPI(ResourceAPI, ABC):
    @abstractmethod
    def get(
        self,
        flow_id: int,
    ) -> OpenMLFlow: ...

    @abstractmethod
    def exists(self, name: str, external_version: str) -> int | bool: ...

    @abstractmethod
    def list(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        uploader: str | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def create(self, flow: OpenMLFlow) -> OpenMLFlow | tuple[OpenMLFlow, Response]: ...

    @abstractmethod
    def delete(self, flow_id: int) -> None | Response: ...
