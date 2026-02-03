from __future__ import annotations

import builtins
from abc import abstractmethod
from typing import Any

from openml.enums import ResourceType

from .base import ResourceAPI


class DatasetAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.DATASET


class TaskAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.TASK


class EvaluationMeasureAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.EVALUATION_MEASURE


class EstimationProcedureAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.ESTIMATION_PROCEDURE

    @abstractmethod
    def list(self) -> list[str]: ...

    @abstractmethod
    def _get_details(self) -> builtins.list[dict[str, Any]]: ...


class EvaluationAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.EVALUATION


class FlowAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.FLOW


class StudyAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.STUDY


class RunAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.RUN


class SetupAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.SETUP
