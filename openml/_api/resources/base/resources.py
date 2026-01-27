from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from openml._api.resources.base import ResourceAPI, ResourceType

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response

    from openml.datasets.dataset import OpenMLDataset
    from openml.flows.flow import OpenMLFlow
    from openml.tasks.task import OpenMLTask


class DatasetsAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.DATASET

    @abstractmethod
    def get(self, dataset_id: int) -> OpenMLDataset | tuple[OpenMLDataset, Response]: ...


class TasksAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.TASK

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
    def publish(self, flow: OpenMLFlow) -> OpenMLFlow | tuple[OpenMLFlow, Response]: ...  # type: ignore[override]

    @abstractmethod
    def delete(self, flow_id: int) -> bool: ...
