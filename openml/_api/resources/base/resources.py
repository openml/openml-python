from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from openml._api.resources.base import ResourceAPI, ResourceType

if TYPE_CHECKING:
    from requests import Response

    from openml.datasets.dataset import OpenMLDataset
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


class EvaluationsAPI(ResourceAPI, ABC):
    @abstractmethod
    def list(
        self,
        limit: int,
        offset: int,
        function: str,
        **kwargs: Any,
    ) -> dict: ...

    @abstractmethod
    def get_users(self, uploader_ids: list[str]) -> dict: ...
