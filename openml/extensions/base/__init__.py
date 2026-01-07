# License: BSD 3-Clause

"""Base classes for OpenML extensions."""

from openml.extensions.base._connector import OpenMLAPIConnector
from openml.extensions.base._executor import ModelExecutor
from openml.extensions.base._serializer import ModelSerializer

__all__ = [
    "ModelExecutor",
    "ModelSerializer",
    "OpenMLAPIConnector",
]
