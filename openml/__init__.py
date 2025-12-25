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

import contextlib
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

    This function provides a unified entry point for publishing various OpenML objects.
    It automatically detects the object type and routes to the appropriate publishing
    mechanism:

    - For OpenML objects (``OpenMLDataset``, ``OpenMLFlow``, ``OpenMLRun``, etc.),
      it directly calls their ``publish()`` method.
    - For external models (e.g., scikit-learn estimators), it uses registered
      extensions to convert them to ``OpenMLFlow`` objects before publishing.

    Parameters
    ----------
    obj : Any
        The object to publish. Can be:
        - An OpenML object (OpenMLDataset, OpenMLFlow, OpenMLRun, OpenMLTask)
        - A machine learning model from a supported framework (e.g., scikit-learn)
    name : str, optional
        Override the default name for the published object.
        If not provided, uses the object's default naming convention.
    tags : Sequence[str], optional
        Additional tags to attach to the published object.
        Will be merged with any existing tags, removing duplicates while
        preserving order.

    Returns
    -------
    Any
        The published object (typically with updated ID and metadata).

    Raises
    ------
    ValueError
        If no extension is registered to handle the provided model type.

    Examples
    --------
    Publishing an OpenML dataset:

    >>> dataset = openml.datasets.get_dataset(61)
    >>> openml.publish(dataset, tags=["example"])

    Publishing a scikit-learn model:

    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(max_depth=5)
    >>> openml.publish(clf, name="MyDecisionTree", tags=["tutorial"])

    Publishing an OpenML flow directly:

    >>> flow = openml.flows.OpenMLFlow(...)
    >>> openml.publish(flow)

    Publishing an OpenML run (after execution with predictions):

    >>> run = openml.runs.OpenMLRun(
    ...     task_id=1, flow_id=100, dataset_id=61,
    ...     data_content=predictions  # predictions from model evaluation
    ... )
    >>> openml.publish(run, tags=["experiment"])

    Notes
    -----
    For external models (e.g., scikit-learn), the corresponding extension must be
    installed (e.g., ``openml-sklearn``). The extension will be automatically imported
    if available.
    """
    # Case 1: Object is already an OpenML entity
    if isinstance(obj, OpenMLBase):
        if tags is not None and hasattr(obj, "tags"):
            existing = list(getattr(obj, "tags", []) or [])
            merged = list(dict.fromkeys([*existing, *tags]))
            obj.tags = merged
        if name is not None and hasattr(obj, "name"):
            obj.name = name
        return obj.publish()

    # Case 2: Object is an external model - use extension registry
    # Attempt to auto-import common extensions
    _ensure_extension_imported(obj)

    extension = extensions.functions.get_extension_by_model(obj, raise_if_no_extension=True)
    if extension is None:  # Defensive check (should not occur with raise_if_no_extension=True)
        raise ValueError("No extension registered to handle the provided object.")
    flow = extension.model_to_flow(obj)

    if name is not None:
        flow.name = name

    if tags is not None:
        existing_tags = list(getattr(flow, "tags", []) or [])
        flow.tags = list(dict.fromkeys([*existing_tags, *tags]))

    return flow.publish()


def _ensure_extension_imported(obj: Any) -> None:
    """Attempt to import the appropriate extension for common frameworks.

    This is a convenience helper to automatically import extensions for
    well-known frameworks, reducing friction for users.

    Parameters
    ----------
    obj : Any
        The object to check.
    """
    obj_module = type(obj).__module__

    # Check for scikit-learn models
    if obj_module.startswith("sklearn"):
        with contextlib.suppress(ImportError):
            import openml_sklearn  # noqa: F401


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
    "OpenMLBenchmarkSuite",
    "OpenMLClassificationTask",
    "OpenMLClusteringTask",
    "OpenMLDataFeature",
    "OpenMLDataset",
    "OpenMLEvaluation",
    "OpenMLFlow",
    "OpenMLLearningCurveTask",
    "OpenMLParameter",
    "OpenMLRegressionTask",
    "OpenMLRun",
    "OpenMLSetup",
    "OpenMLSplit",
    "OpenMLStudy",
    "OpenMLSupervisedTask",
    "OpenMLTask",
    "__version__",
    "_api_calls",
    "config",
    "datasets",
    "evaluations",
    "exceptions",
    "extensions",
    "flows",
    "publish",
    "runs",
    "setups",
    "study",
    "tasks",
    "utils",
]
