# License: BSD 3-Clause

"""Extension registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openml.exceptions import PyOpenMLError
from openml.extensions.sklearn import SklearnAPIConnector

if TYPE_CHECKING:
    from openml.extensions.base import OpenMLAPIConnector

API_CONNECTOR_REGISTRY: list[type[OpenMLAPIConnector]] = [
    SklearnAPIConnector,
]


def resolve_api_connector(estimator: Any) -> OpenMLAPIConnector:
    """
    Identify and return the appropriate OpenML API connector for a given estimator.

    This function iterates through the global ``API_CONNECTOR_REGISTRY`` to find
    a connector class that supports the provided estimator instance or OpenML flow.
    If a matching connector is found, it is instantiated and returned.

    Parameters
    ----------
    estimator : Any
        The estimator instance (e.g., a scikit-learn estimator) or OpenML flow for
        which an API connector is required.

    Returns
    -------
    OpenMLAPIConnector
        An instance of the matching API connector.

    Raises
    ------
    OpenMLException
        If no connector is found in the registry that supports the provided
        model, or if multiple connectors in the registry claim support for
        the provided model.
    """
    for connector_cls in API_CONNECTOR_REGISTRY:
        if connector_cls.supports(estimator):
            return connector_cls()

    raise PyOpenMLError("No OpenML API connector supports this estimator.")
