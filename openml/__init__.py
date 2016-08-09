"""
The OpenML module implements a python interface to
`OpenML <http://openml.org>`_, a collaborative platform for machine learning.
OpenML can be used to

* store, download and analyze datasets
* make experiments and their results (e.g. models, predictions)
  accesible and reproducible for everybody
* analyze experiments (uploaded by you and other collaborators) and conduct
  meta studies

In particular, this module implemts a python interface for the
`OpenML REST API <http://openml.org/guide#!rest_services>`_
(`REST on wikipedia
<http://en.wikipedia.org/wiki/Representational_state_transfer>`_).
"""
from . import config

from .datasets import OpenMLDataset
from . import datasets
from . import runs
from . import flows
from .runs import OpenMLRun
from .tasks import OpenMLTask, OpenMLSplit
from .flows import OpenMLFlow


__version__ = "0.2.1"

__all__ = ['OpenMLDataset', 'OpenMLRun', 'OpenMLSplit',
           'datasets', 'OpenMLTask', 'OpenMLFlow', 'config', 'runs', 'flows']
