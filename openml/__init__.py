"""
The OpenML module implements a python interface to
`OpenML <https://www.openml.org>`_, a collaborative platform for machine
learning. OpenML can be used to

* store, download and analyze datasets
* make experiments and their results (e.g. models, predictions)
  accesible and reproducible for everybody
* analyze experiments (uploaded by you and other collaborators) and conduct
  meta studies

In particular, this module implemts a python interface for the
`OpenML REST API <https://www.openml.org/guide#!rest_services>`_
(`REST on wikipedia
<http://en.wikipedia.org/wiki/Representational_state_transfer>`_).
"""
from . import config

from .datasets import OpenMLDataset, OpenMLDataFeature
from . import datasets
from . import tasks
from . import runs
from . import flows
from . import setups
from . import study
from . import evaluations
from . import utils
from .runs import OpenMLRun
from .tasks import OpenMLTask, OpenMLSplit
from .flows import OpenMLFlow
from .evaluations import OpenMLEvaluation

from .__version__ import __version__


def populate_cache(task_ids=None, dataset_ids=None, flow_ids=None,
                   run_ids=None):
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


__all__ = ['OpenMLDataset', 'OpenMLDataFeature', 'OpenMLRun',
           'OpenMLSplit', 'OpenMLEvaluation', 'OpenMLSetup',
           'OpenMLTask', 'OpenMLFlow', 'datasets', 'evaluations',
           'config', 'runs', 'flows', 'tasks', 'setups']
