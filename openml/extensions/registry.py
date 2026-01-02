# License: BSD 3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openml_sklearn import SklearnExtension

from openml.exceptions import OpenMLException

if TYPE_CHECKING:
    from openml.extensions.connectors import OpenMLAPIConnector

API_CONNECTOR_REGISTRY: list[type[OpenMLAPIConnector]] = [
    SklearnExtension,  # TODO: I need to refactor SklearnExtension
]


def resolve_api_connector(estimator: Any) -> OpenMLAPIConnector:
    """
    Identifies and returns the appropriate OpenML API connector for a given estimator.

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
    candidates = [
        connector for connector in API_CONNECTOR_REGISTRY if connector.supports(estimator)
    ]

    if not candidates:
        raise OpenMLException("No OpenML API connector found for this estimator.")

    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise OpenMLException(
            "Multiple API connectors match this estimator:\n" + "\n".join(f"- {n}" for n in names)
        )

    return candidates[0]()
