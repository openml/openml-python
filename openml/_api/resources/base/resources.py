from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from openml._api.resources.base import ResourceAPI, ResourceType

if TYPE_CHECKING:
    import pandas as pd
    from requests import Response

    from openml.datasets.dataset import OpenMLDataset
    from openml.runs.run import OpenMLRun
    from openml.tasks.task import OpenMLTask, TaskType


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


class RunsAPI(ResourceAPI, ABC):
    resource_type: ResourceType = ResourceType.RUN

    @abstractmethod
    def get(
        self, run_id: int, *, use_cache: bool = True, reset_cache: bool = False
    ) -> OpenMLRun: ...

    @abstractmethod
    def list(  # type: ignore[valid-type]  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        ids: builtins.list[int] | None = None,
        task: builtins.list[int] | None = None,
        setup: builtins.list[int] | None = None,
        flow: builtins.list[int] | None = None,
        uploader: builtins.list[int] | None = None,
        study: int | None = None,
        tag: str | None = None,
        display_errors: bool = False,
        task_type: TaskType | int | None = None,
    ) -> pd.DataFrame: ...
