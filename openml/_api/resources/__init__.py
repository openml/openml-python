from openml._api.resources.base.fallback import FallbackProxy
from openml._api.resources.datasets import DatasetsV1, DatasetsV2
from openml._api.resources.evaluations import EvaluationsV1, EvaluationsV2
from openml._api.resources.tasks import TasksV1, TasksV2

__all__ = [
    "DatasetsV1",
    "DatasetsV2",
    "EvaluationsV1",
    "EvaluationsV2",
    "FallbackProxy",
    "TasksV1",
    "TasksV2",
]
