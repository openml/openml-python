from __future__ import annotations

from enum import Enum


class APIVersion(str, Enum):
    """Supported OpenML API versions."""

    V1 = "v1"
    V2 = "v2"


class ResourceType(str, Enum):
    """Canonical resource types exposed by the OpenML API."""

    DATASET = "dataset"
    TASK = "task"
    TASK_TYPE = "task_type"
    EVALUATION_MEASURE = "evaluation_measure"
    ESTIMATION_PROCEDURE = "estimation_procedure"
    EVALUATION = "evaluation"
    FLOW = "flow"
    STUDY = "study"
    RUN = "run"
    SETUP = "setup"
    USER = "user"


class RetryPolicy(str, Enum):
    """Retry behavior for failed API requests."""

    HUMAN = "human"
    ROBOT = "robot"
