from openml._api.resources.base.fallback import FallbackProxy
from openml._api.resources.datasets import DatasetsV1, DatasetsV2
from openml._api.resources.setups import SetupsV1, SetupsV2
from openml._api.resources.tasks import TasksV1, TasksV2

__all__ = [
    "DatasetsV1",
    "DatasetsV2",
    "FallbackProxy",
    "SetupsV1",
    "SetupsV2",
    "TasksV1",
    "TasksV2",
]
