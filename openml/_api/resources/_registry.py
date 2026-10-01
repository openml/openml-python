from __future__ import annotations

from typing import TYPE_CHECKING

from openml.enums import APIVersion, ResourceType

from .dataset import DatasetV1API, DatasetV2API
from .estimation_procedure import (
    EstimationProcedureV1API,
    EstimationProcedureV2API,
)
from .evaluation import EvaluationV1API, EvaluationV2API
from .evaluation_measure import EvaluationMeasureV1API, EvaluationMeasureV2API
from .flow import FlowV1API, FlowV2API
from .run import RunV1API, RunV2API
from .setup import SetupV1API, SetupV2API
from .study import StudyV1API, StudyV2API
from .task import TaskV1API, TaskV2API

if TYPE_CHECKING:
    from .base import ResourceAPI

API_REGISTRY: dict[
    APIVersion,
    dict[ResourceType, type[ResourceAPI]],
] = {
    APIVersion.V1: {
        ResourceType.DATASET: DatasetV1API,
        ResourceType.TASK: TaskV1API,
        ResourceType.EVALUATION_MEASURE: EvaluationMeasureV1API,
        ResourceType.ESTIMATION_PROCEDURE: EstimationProcedureV1API,
        ResourceType.EVALUATION: EvaluationV1API,
        ResourceType.FLOW: FlowV1API,
        ResourceType.STUDY: StudyV1API,
        ResourceType.RUN: RunV1API,
        ResourceType.SETUP: SetupV1API,
    },
    APIVersion.V2: {
        ResourceType.DATASET: DatasetV2API,
        ResourceType.TASK: TaskV2API,
        ResourceType.EVALUATION_MEASURE: EvaluationMeasureV2API,
        ResourceType.ESTIMATION_PROCEDURE: EstimationProcedureV2API,
        ResourceType.EVALUATION: EvaluationV2API,
        ResourceType.FLOW: FlowV2API,
        ResourceType.STUDY: StudyV2API,
        ResourceType.RUN: RunV2API,
        ResourceType.SETUP: SetupV2API,
    },
}
