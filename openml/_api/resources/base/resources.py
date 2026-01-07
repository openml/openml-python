from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

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


class EstimationProceduresAPI(ResourceAPI, ABC):
    @abstractmethod
    def list(self) -> list[str]: ...
