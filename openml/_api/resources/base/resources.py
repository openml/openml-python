from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from openml._api.resources.base import ResourceAPI, ResourceType

if TYPE_CHECKING:
    from requests import Response

    from openml.datasets.dataset import OpenMLDataset
    from openml.setups.setup import OpenMLSetup
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


class SetupsAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.SETUP

    @abstractmethod
    def list(
        self,
        limit: int,
        offset: int,
        *,
        setup: Iterable[int] | None = None,
        flow: int | None = None,
        tag: str | None = None,
    ) -> list[OpenMLSetup]: ...

    @abstractmethod
    def _create_setup(self, result_dict: dict) -> OpenMLSetup: ...

    @abstractmethod
    def get(self, setup_id: int) -> tuple[str, OpenMLSetup]: ...

    @abstractmethod
    def exists(self, file_elements: dict[str, Any]) -> int: ...
