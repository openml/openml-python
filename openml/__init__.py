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

_LIST_DISPATCH: ListDispatcher = {
    "dataset": datasets.functions.list_datasets,
    "task": tasks.functions.list_tasks,
    "flow": flows.functions.list_flows,
    "run": runs.functions.list_runs,
}

_GET_DISPATCH: GetDispatcher = {
    "dataset": datasets.functions.get_dataset,
    "task": tasks.functions.get_task,
    "flow": flows.functions.get_flow,
    "run": runs.functions.get_run,
}


def list_all(object_type: str, /, **kwargs: Any) -> Any:
    """List OpenML objects by type (e.g., datasets, tasks, flows, runs).

    This is a convenience dispatcher that forwards to the existing type-specific
    ``list_*`` functions. Existing imports remain available for backward compatibility.

    Parameters
    ----------
    object_type : str
        The type of object to list. Must be one of 'dataset', 'task', 'flow', 'run'.
    **kwargs : Any
        Additional arguments passed to the underlying list function.

    Returns
    -------
    Any
        The result from the type-specific list function (typically a DataFrame).

    Raises
    ------
    ValueError
        If object_type is not one of the supported types.
    """
    if not isinstance(object_type, str):
        raise TypeError(f"object_type must be a string, got {type(object_type).__name__}")

    func = _LIST_DISPATCH.get(object_type.lower())
    if func is None:
        valid_types = ", ".join(repr(k) for k in _LIST_DISPATCH)
        raise ValueError(
            f"Unsupported object_type {object_type!r}; expected one of {valid_types}.",
        )

    return func(**kwargs)


def get(identifier: int | str, *, object_type: str = "dataset", **kwargs: Any) -> Any:
    """Get an OpenML object by identifier.

    Parameters
    ----------
    identifier : int | str
        The ID or name of the object to retrieve.
    object_type : str, default="dataset"
        The type of object to get. Must be one of 'dataset', 'task', 'flow', 'run'.
    **kwargs : Any
        Additional arguments passed to the underlying get function.

    Returns
    -------
    Any
        The requested OpenML object.

    Raises
    ------
    ValueError
        If object_type is not one of the supported types.

    Examples
    --------
    >>> openml.get(61)  # Get dataset 61 (default object_type="dataset")
    >>> openml.get("Fashion-MNIST")  # Get dataset by name
    >>> openml.get(31, object_type="task")  # Get task 31
    >>> openml.get(10, object_type="flow")  # Get flow 10
    >>> openml.get(20, object_type="run")  # Get run 20
    """
    if not isinstance(object_type, str):
        raise TypeError(f"object_type must be a string, got {type(object_type).__name__}")

    func = _GET_DISPATCH.get(object_type.lower())
    if func is None:
        valid_types = ", ".join(repr(k) for k in _GET_DISPATCH)
        raise ValueError(
            f"Unsupported object_type {object_type!r}; expected one of {valid_types}.",
        )

    return func(identifier, **kwargs)


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
    "get",
    "list_all",
]
