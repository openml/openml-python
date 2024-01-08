# License: BSD 3-Clause

from .functions import (
    create_task,
    delete_task,
    get_task,
    get_tasks,
    list_tasks,
)
from .split import OpenMLSplit
from .task import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    OpenMLTask,
    TaskType,
)

__all__ = [
    "OpenMLTask",
    "OpenMLSupervisedTask",
    "OpenMLClusteringTask",
    "OpenMLRegressionTask",
    "OpenMLClassificationTask",
    "OpenMLLearningCurveTask",
    "create_task",
    "get_task",
    "get_tasks",
    "list_tasks",
    "OpenMLSplit",
    "TaskType",
    "delete_task",
]
