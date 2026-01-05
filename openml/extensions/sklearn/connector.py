# License: BSD 3-Clause

"""Connector for the Scikit-learn extension."""

from __future__ import annotations

from openml.extensions.base import OpenMLAPIConnector
from openml.extensions.sklearn.executor import SklearnExecutor
from openml.extensions.sklearn.serializer import SklearnSerializer
from openml.flows import OpenMLFlow


class SklearnAPIConnector(OpenMLAPIConnector):
    """
    Connector for the Scikit-learn extension.

    This class provides the interface to connect Scikit-learn models and flows
    to the OpenML API, handling both serialization and execution compatibility checks.
    """

    def serializer(self) -> SklearnSerializer:
        """
        Return the serializer for Scikit-learn estimators.

        Returns
        -------
        SklearnSerializer
            The serializer instance capable of handling Scikit-learn estimator.
        """
        return SklearnSerializer()

    def executor(self) -> SklearnExecutor:
        """
        Return the executor for Scikit-learn estimators.

        Returns
        -------
        SklearnExecutor
            The executor instance capable of running Scikit-learn estimators.
        """
        return SklearnExecutor()

    @classmethod
    def supports(cls, estimator) -> bool:
        """
        Check if this connector supports the given model or flow.

        Parameters
        ----------
        estimator : Any or OpenMLFlow
            The Scikit-learn estimator instance or OpenMLFlow object.

        Returns
        -------
        bool
            True if both the serializer and executor can handle the provided
            estimator or flow, False otherwise.
        """
        serializer = SklearnSerializer()
        SklearnExecutor()

        if isinstance(estimator, OpenMLFlow):
            support = serializer.can_handle_flow(estimator)

        else:
            support = serializer.can_handle_model(estimator)

        return support
