from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    from openml.estimation_procedures.estimation_procedure import OpenMLEstimationProcedure


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

    @abstractmethod
    def list(self) -> list[OpenMLEstimationProcedure]: ...


class EvaluationAPI(ResourceAPI):
    """Abstract API interface for evaluation resources."""

    resource_type: ResourceType = ResourceType.EVALUATION


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
