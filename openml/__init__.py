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

import builtins
from typing import Any, Callable, Dict

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

ListDispatcher = Dict[str, Callable[..., Any]]
GetDispatcher = Dict[str, Callable[..., Any]]


def list(object_type: str, /, **kwargs: Any) -> Any:  # noqa: A001
    """List OpenML objects by type (e.g., datasets, tasks, flows, runs).

    This is a convenience dispatcher that forwards to the existing type-specific
    ``list_*`` functions. Existing imports remain available for backward compatibility.
    """
    dispatch: ListDispatcher = {
        "dataset": datasets.functions.list_datasets,
        "task": tasks.functions.list_tasks,
        "flow": flows.functions.list_flows,
        "run": runs.functions.list_runs,
    }

    try:
        func = dispatch[object_type.lower()]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            "Unsupported object_type for list; expected one of 'dataset', 'task', 'flow', 'run'.",
        ) from exc

    return func(**kwargs)


def get(object_type_or_name: Any, identifier: Any | None = None, /, **kwargs: Any) -> Any:
    """Get an OpenML object by type and identifier, or a dataset by name.

    Examples
    --------
    openml.get("dataset", 61)
    openml.get("dataset", "Fashion-MNIST")
    openml.get("task", 31)
    openml.get("flow", 10)
    openml.get("run", 20)
    openml.get("Fashion-MNIST")  # dataset lookup by name (no type specified)
    """
    # Single-argument shortcut: treat string without type as dataset lookup.
    if identifier is None:
        if isinstance(object_type_or_name, str):
            return datasets.functions.get_dataset(object_type_or_name, **kwargs)
        raise ValueError("Please provide an object_type when identifier is not provided.")

    object_type = str(object_type_or_name).lower()
    dispatch: GetDispatcher = {
        "dataset": datasets.functions.get_dataset,
        "task": tasks.functions.get_task,
        "flow": flows.functions.get_flow,
        "run": runs.functions.get_run,
    }

    try:
        func = dispatch[object_type]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            "Unsupported object_type for get; expected one of 'dataset', 'task', 'flow', 'run'.",
        ) from exc

    return func(identifier, **kwargs)


def populate_cache(
    task_ids: builtins.list[int] | None = None,
    dataset_ids: builtins.list[int | str] | None = None,
    flow_ids: builtins.list[int] | None = None,
    run_ids: builtins.list[int] | None = None,
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
    "list",
    "get",
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
