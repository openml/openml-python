# License: BSD 3-Clause

"""Base class for estimator serializors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openml.flows import OpenMLFlow


class ModelSerializer(ABC):
    """Handle the conversion between estimator instances and OpenML Flows."""

    @classmethod
    @abstractmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model flow can be handled by this extension.

        This is typically done by checking the type of the model, or the package it belongs to.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """

    @abstractmethod
    def model_to_flow(self, model: Any) -> OpenMLFlow:
        """Transform a model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """

    @abstractmethod
    def flow_to_model(
        self,
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
        Any
        """

    @abstractmethod
    def get_version_information(self) -> list[str]:
        """Return dependency and version information."""

    @abstractmethod
    def obtain_parameter_values(
        self,
        flow: OpenMLFlow,
        model: Any = None,
    ) -> list[dict[str, Any]]:
        """Extracts all parameter settings required for the flow from the model.

        If no explicit model is provided, the parameters will be extracted from `flow.model`
        instead.

        Parameters
        ----------
        flow : OpenMLFlow
            OpenMLFlow object (containing flow ids, i.e., it has to be downloaded from the server)

        model: Any, optional (default=None)
            The model from which to obtain the parameter values. Must match the flow signature.
            If None, use the model specified in ``OpenMLFlow.model``.

        Returns
        -------
        list
            A list of dicts, where each dict has the following entries:
            - ``oml:name`` : str: The OpenML parameter name
            - ``oml:value`` : mixed: A representation of the parameter value
            - ``oml:component`` : int: flow id to which the parameter belongs
        """
