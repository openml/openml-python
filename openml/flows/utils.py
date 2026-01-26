# License: BSD 3-Clause

"""Utility functions for OpenML extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openml.extensions.registry import resolve_api_connector

if TYPE_CHECKING:
    from openml.flows import OpenMLFlow


def flow_to_estimator(
    flow: OpenMLFlow,
    initialize_with_defaults: bool = False,  # noqa: FBT002
    strict_version: bool = True,  # noqa: FBT002
) -> Any:
    """Instantiate a model from the flow representation.

    Parameters
    ----------
    flow : OpenMLFlow

    initialize_with_defaults : bool, optional (default=False)
        If this flag is set, the hyperparameter values of flows will be
        ignored and a flow with its defaults is returned.

    strict_version : bool, default=True
        Whether to fail if version requirements are not fulfilled.

    Returns
    -------
    estimator_instance : Any
        The corresponding estimator instance.
    """
    connector = resolve_api_connector(flow)
    return connector.serializer().flow_to_model(
        flow,
        initialize_with_defaults=initialize_with_defaults,
        strict_version=strict_version,
    )


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
