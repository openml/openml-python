from __future__ import annotations

import builtins
from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    import pandas as pd

    from openml.estimation_procedures import OpenMLEstimationProcedure
    from openml.evaluations.evaluation import OpenMLEvaluation
    from openml.flows.flow import OpenMLFlow
    from openml.setups.setup import OpenMLSetup
    from openml.tasks.task import OpenMLTask, TaskType


class DatasetAPI(ResourceAPI):
    """Abstract API interface for dataset resources."""

    resource_type: ResourceType = ResourceType.DATASET


class TaskAPI(ResourceAPI):
    """Abstract API interface for task resources."""

    resource_type: ResourceType = ResourceType.TASK

    @abstractmethod
    def get(
        self,
        task_id: int,
    ) -> OpenMLTask:
        """
        API v1:
            GET /task/{task_id}

        API v2:
            GET /tasks/{task_id}
        """
        ...

    @abstractmethod
    def supports_download_splits(self) -> bool:
        """Return whether the task API implementation supports split downloads."""
        ...

    # Task listing (V1 only)
    @abstractmethod
    def list(
        self,
        limit: int,
        offset: int,
        task_type: TaskType | int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        List tasks with filters.

        API v1:
            GET /task/list

        API v2:
            Not available.

        Returns
        -------
        pandas.DataFrame
        """
        ...


class EvaluationMeasureAPI(ResourceAPI):
    """Abstract API interface for evaluation measure resources."""

    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE

    @abstractmethod
    def list(self) -> list[str]: ...


class EstimationProcedureAPI(ResourceAPI):
    """Abstract API interface for estimation procedure resources."""

    resource_type: ResourceType = ResourceType.ESTIMATION_PROCEDURE

    @abstractmethod
    def list(self) -> list[OpenMLEstimationProcedure]: ...


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


class SetupAPI(ResourceAPI):
    """Abstract API interface for setup resources."""

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
    def get(self, setup_id: int) -> OpenMLSetup: ...

    @abstractmethod
    def exists(
        self,
        flow: OpenMLFlow,
        param_settings: builtins.list[dict[str, Any]],
    ) -> int | bool: ...
