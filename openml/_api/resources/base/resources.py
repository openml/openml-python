from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    import pandas as pd

    from openml.evaluations import OpenMLEvaluation
    from openml.runs.run import OpenMLRun
    from openml.tasks.task import TaskType


class DatasetAPI(ResourceAPI):
    """Abstract API interface for dataset resources."""

    resource_type: ResourceType = ResourceType.DATASET


class TaskAPI(ResourceAPI):
    """Abstract API interface for task resources."""

    resource_type: ResourceType = ResourceType.TASK


class EvaluationMeasureAPI(ResourceAPI):
    """Abstract API interface for evaluation measure resources."""

    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE


class EstimationProcedureAPI(ResourceAPI):
    """Abstract API interface for estimation procedure resources."""

    resource_type: ResourceType = ResourceType.ESTIMATION_PROCEDURE


class EvaluationAPI(ResourceAPI):
    """Abstract API interface for evaluation resources."""

    resource_type: ResourceType = ResourceType.EVALUATION

    @abstractmethod
    def list(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        *,
        function: str,
        tasks: list | None = None,
        setups: list | None = None,
        flows: list | None = None,
        runs: list | None = None,
        uploaders: list | None = None,
        study: int | None = None,
        sort_order: str | None = None,
        **kwargs: Any,
    ) -> list[OpenMLEvaluation]: ...


class FlowAPI(ResourceAPI):
    """Abstract API interface for flow resources."""

    resource_type: ResourceType = ResourceType.FLOW


class StudyAPI(ResourceAPI):
    """Abstract API interface for study resources."""

    resource_type: ResourceType = ResourceType.STUDY


class RunAPI(ResourceAPI):
    """Abstract API interface for run resources."""

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
    """Abstract API interface for setup resources."""

    resource_type: ResourceType = ResourceType.SETUP
