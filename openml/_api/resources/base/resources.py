from __future__ import annotations

import builtins
from abc import abstractmethod
from typing import TYPE_CHECKING

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    import pandas as pd

    from openml.runs.run import OpenMLRun
    from openml.tasks.task import TaskType


class DatasetAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.DATASET


class TaskAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.TASK


class EvaluationMeasureAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE


class EstimationProcedureAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.ESTIMATION_PROCEDURE


class EvaluationAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.EVALUATION


class FlowAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.FLOW


class StudyAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.STUDY


class RunAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.RUN

    @abstractmethod
    def get(
        self,
        run_id: int,
        *,
        reset_cache: bool = False,
    ) -> OpenMLRun: ...

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


class SetupAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.SETUP
