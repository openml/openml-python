"""OpenML API dispatchers for unified get/list operations."""

# License: BSD 3-Clause
from __future__ import annotations

from typing import Any, Callable, Dict

from .datasets import get_dataset, list_datasets
from .flows import get_flow, list_flows
from .runs import get_run, list_runs
from .tasks import get_task, list_tasks

ListDispatcher = Dict[str, Callable[..., Any]]
GetDispatcher = Dict[str, Callable[..., Any]]

_LIST_DISPATCH: ListDispatcher = {
    "dataset": list_datasets,
    "task": list_tasks,
    "flow": list_flows,
    "run": list_runs,
}

_GET_DISPATCH: GetDispatcher = {
    "dataset": get_dataset,
    "task": get_task,
    "flow": get_flow,
    "run": get_run,
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
