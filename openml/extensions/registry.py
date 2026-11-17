# License: BSD 3-Clause

"""Extension registries for serializers and executors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openml.exceptions import PyOpenMLError

if TYPE_CHECKING:
    from openml.extensions.base import ModelExecutor, ModelSerializer


SERIALIZER_REGISTRY: list[type[ModelSerializer]] = []
EXECUTOR_REGISTRY: list[type[ModelExecutor]] = []


def register_serializer(cls: type[ModelSerializer]) -> type[ModelSerializer]:
    """Register a serializer class."""
    SERIALIZER_REGISTRY.append(cls)
    return cls


def register_executor(cls: type[ModelExecutor]) -> type[ModelExecutor]:
    """Register an executor class."""
    EXECUTOR_REGISTRY.append(cls)
    return cls


def resolve_serializer(estimator: Any) -> ModelSerializer:
    """
    Identify and return the appropriate serializer for a given estimator.

    Parameters
    ----------
    estimator : Any
        The estimator instance (e.g., sklearn estimator, sktime estimator).

    Returns
    -------
    ModelSerializer
        An instance of the matching serializer.

    Raises
    ------
    PyOpenMLError
        If no serializer supports the estimator or if multiple serializers match.
    """
    matches = [
        serializer_cls
        for serializer_cls in SERIALIZER_REGISTRY
        if serializer_cls.can_handle_model(estimator)
    ]

    if len(matches) == 1:
        return matches[0]()

    if len(matches) > 1:
        raise PyOpenMLError("Multiple serializers support this estimator.")

    raise PyOpenMLError("No serializer supports this estimator.")


def resolve_executor(estimator: Any) -> ModelExecutor:
    """
    Identify and return the appropriate executor for a given estimator.

    Parameters
    ----------
    estimator : Any
        The estimator instance.

    Returns
    -------
    ModelExecutor
        An instance of the matching executor.

    Raises
    ------
    PyOpenMLError
        If no executor supports the estimator or if multiple executors match.
    """
    matches = [
        executor_cls
        for executor_cls in EXECUTOR_REGISTRY
        if executor_cls.can_handle_model(estimator)
    ]

    if len(matches) == 1:
        return matches[0]()

    if len(matches) > 1:
        raise PyOpenMLError("Multiple executors support this estimator.")

    raise PyOpenMLError("No executor supports this estimator.")
