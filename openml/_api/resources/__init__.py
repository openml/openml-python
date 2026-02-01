from openml._api.resources.base.fallback import FallbackProxy
from openml._api.resources.dataset import DatasetV1API, DatasetV2API
from openml._api.resources.estimation_procedure import (
    EstimationProcedureV1API,
    EstimationProcedureV2API,
)
from openml._api.resources.evaluation import EvaluationV1API, EvaluationV2API
from openml._api.resources.evaluation_measure import EvaluationMeasureV1API, EvaluationMeasureV2API
from openml._api.resources.flow import FlowV1API, FlowV2API
from openml._api.resources.run import RunV1API, RunV2API
from openml._api.resources.setup import SetupV1API, SetupV2API
from openml._api.resources.study import StudyV1API, StudyV2API
from openml._api.resources.task import TaskV1API, TaskV2API

__all__ = [
    "DatasetV1API",
    "DatasetV2API",
    "EstimationProcedureV1API",
    "EstimationProcedureV2API",
    "EvaluationMeasureV1API",
    "EvaluationMeasureV2API",
    "EvaluationV1API",
    "EvaluationV2API",
    "FallbackProxy",
    "FlowV1API",
    "FlowV2API",
    "RunV1API",
    "RunV2API",
    "SetupV1API",
    "SetupV2API",
    "StudyV1API",
    "StudyV2API",
    "TaskV1API",
    "TaskV2API",
]
