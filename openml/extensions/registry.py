# License: BSD 3-Clause

"""Extension registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openml_sklearn import SklearnExtension

from openml.exceptions import PyOpenMLError

if TYPE_CHECKING:
    from openml.extensions.connectors import OpenMLAPIConnector

API_CONNECTOR_REGISTRY: list[type[OpenMLAPIConnector]] = [
    SklearnExtension,  # TODO: I need to refactor SklearnExtension
]


def resolve_api_connector(estimator: Any) -> OpenMLAPIConnector:
    """
    Identify and return the appropriate OpenML API connector for a given estimator.

    This function iterates through the global ``API_CONNECTOR_REGISTRY`` to find
    a connector class that supports the provided estimator object. If exactly one
    matching connector is found, it is instantiated and returned.

    Parameters
    ----------
    estimator : Any
        The estimator object (e.g., a scikit-learn estimator) for which an API
        connector is required.

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
