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
]

# Load the scikit-learn extension by default
# TODO(eddiebergman): Not sure why this is at the bottom of the file
import openml.extensions.sklearn  # noqa: E402, F401
