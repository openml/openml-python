from openml._api.resources.base.fallback import FallbackProxy
from openml._api.resources.datasets import DatasetsV1, DatasetsV2
from openml._api.resources.flows import FlowsV1, FlowsV2
from openml._api.resources.tasks import TasksV1, TasksV2

__all__ = [
    "DatasetsV1",
    "DatasetsV2",
    "FallbackProxy",
    "FlowsV1",
    "FlowsV2",
    "TasksV1",
    "TasksV2",
]
