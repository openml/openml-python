from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from openml.enums import ResourceType

from .base import ResourceAPI

if TYPE_CHECKING:
    import pandas as pd

    from openml.flows.flow import OpenMLFlow


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

    @abstractmethod
    def get(self, flow_id: int, *, reset_cache: bool = False) -> OpenMLFlow: ...

    @abstractmethod
    def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        uploader: str | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def exists(self, name: str, external_version: str) -> int | bool: ...


class StudyAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.STUDY


class RunAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.RUN


class SetupAPI(ResourceAPI):
    resource_type: ResourceType = ResourceType.SETUP
