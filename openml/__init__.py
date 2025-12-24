"""
The OpenML module implements a python interface to
`OpenML <https://www.openml.org>`_, a collaborative platform for machine
learning. OpenML can be used to

* store, download and analyze datasets
* make experiments and their results (e.g. models, predictions)
  accesible and reproducible for everybody
* analyze experiments (uploaded by you and other collaborators) and conduct
  meta studies

In particular, this module implements a python interface for the
`OpenML REST API <https://www.openml.org/guide#!rest_services>`_
(`REST on wikipedia
<https://en.wikipedia.org/wiki/Representational_state_transfer>`_).
"""

# License: BSD 3-Clause
from __future__ import annotations

from typing import Any, Sequence

from . import (
    _api_calls,
    config,
    datasets,
    evaluations,
    exceptions,
    extensions,
    flows,
    runs,
    setups,
    study,
    tasks,
    utils,
)
from .__version__ import __version__
from .base import OpenMLBase
from .datasets import OpenMLDataFeature, OpenMLDataset
from .evaluations import OpenMLEvaluation
from .flows import OpenMLFlow
from .runs import OpenMLRun
from .setups import OpenMLParameter, OpenMLSetup
from .study import OpenMLBenchmarkSuite, OpenMLStudy
from .tasks import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLSplit,
    OpenMLSupervisedTask,
    OpenMLTask,
)


def publish(obj: Any, *, name: str | None = None, tags: Sequence[str] | None = None) -> Any:
    """Publish a common object (flow/model/run/dataset) with minimal friction.

    If ``obj`` is already an OpenML object (``OpenMLBase``) it will call its ``publish`` method.
    Otherwise it looks for a registered extension (e.g., scikit-learn) to convert the object
    into an ``OpenMLFlow`` and publish it.
    """
    if isinstance(obj, OpenMLBase):
        if tags is not None and hasattr(obj, "tags"):
            existing = list(getattr(obj, "tags", []) or [])
            merged = list(dict.fromkeys([*existing, *tags]))
            obj.tags = merged
        if name is not None and hasattr(obj, "name"):
            obj.name = name
        return obj.publish()

    extension = extensions.functions.get_extension_by_model(obj, raise_if_no_extension=True)
    if extension is None:  # defensive; should not happen with raise_if_no_extension=True
        raise ValueError("No extension registered to handle the provided object.")
    flow = extension.model_to_flow(obj)

    if name is not None:
        flow.name = name

    if tags is not None:
        existing_tags = list(getattr(flow, "tags", []) or [])
        flow.tags = list(dict.fromkeys([*existing_tags, *tags]))

    return flow.publish()


def populate_cache(
    task_ids: list[int] | None = None,
    dataset_ids: list[int | str] | None = None,
    flow_ids: list[int] | None = None,
    run_ids: list[int] | None = None,
) -> None:
    """
    Populate a cache for offline and parallel usage of the OpenML connector.

    Parameters
    ----------
    task_ids : iterable

    dataset_ids : iterable

    flow_ids : iterable

    run_ids : iterable

    Returns
    -------
    None
    """
    if task_ids is not None:
        for task_id in task_ids:
            tasks.functions.get_task(task_id)

    if dataset_ids is not None:
        for dataset_id in dataset_ids:
            datasets.functions.get_dataset(dataset_id)

    if flow_ids is not None:
        for flow_id in flow_ids:
            flows.functions.get_flow(flow_id)

    if run_ids is not None:
        for run_id in run_ids:
            runs.functions.get_run(run_id)


__all__ = [
    "publish",
    "OpenMLDataset",
    "OpenMLDataFeature",
    "OpenMLRun",
    "OpenMLSplit",
    "OpenMLEvaluation",
    "OpenMLSetup",
    "OpenMLParameter",
    "OpenMLTask",
    "OpenMLSupervisedTask",
    "OpenMLClusteringTask",
    "OpenMLLearningCurveTask",
    "OpenMLRegressionTask",
    "OpenMLClassificationTask",
    "OpenMLFlow",
    "OpenMLStudy",
    "OpenMLBenchmarkSuite",
    "datasets",
    "evaluations",
    "exceptions",
    "extensions",
    "config",
    "runs",
    "flows",
    "tasks",
    "setups",
    "study",
    "utils",
    "_api_calls",
    "__version__",
]
