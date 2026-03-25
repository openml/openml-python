from __future__ import annotations

import builtins
from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    from openml.evaluations import OpenMLEvaluation
    from openml.flows.flow import OpenMLFlow
    from openml.setups.setup import OpenMLSetup


class DatasetAPI(ResourceAPI):
    """Abstract API interface for dataset resources."""

    resource_type: ResourceType = ResourceType.DATASET


class TaskAPI(ResourceAPI):
    """Abstract API interface for task resources."""

    resource_type: ResourceType = ResourceType.TASK


class EvaluationMeasureAPI(ResourceAPI):
    """Abstract API interface for evaluation measure resources."""

    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE

    @abstractmethod
    def list(self) -> list[str]: ...


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
