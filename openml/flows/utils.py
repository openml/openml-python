# License: BSD 3-Clause

"""Utility functions for OpenML extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openml.extensions.registry import resolve_api_connector

if TYPE_CHECKING:
    from openml.flows import OpenMLFlow


def flow_to_estimator(flow: OpenMLFlow) -> Any:
    """Convert an OpenML flow to an estimator instance.

    Parameters
    ----------
    flow : openml.flows.OpenMLFlow
        The OpenML flow to convert.

    Returns
    -------
    estimator_instance : Any
        The corresponding estimator instance.
    """
    connector = resolve_api_connector(flow)
    return connector.serializer().flow_to_model(flow)


def estimator_to_flow(estimator_instance: Any) -> OpenMLFlow:
    """Convert an estimator instance to an OpenML flow.

    Parameters
    ----------
    estimator_instance : Any
        The estimator instance to convert.

    Returns
    -------
    flow : openml.flows.OpenMLFlow
        The corresponding OpenML flow.
    """
    connector = resolve_api_connector(estimator_instance)
    return connector.serializer().model_to_flow(estimator_instance)
