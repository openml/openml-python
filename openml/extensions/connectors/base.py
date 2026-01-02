# License: BSD 3-Clause

"""Base class for OpenML API connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openml.extensions.execution import ModelExecutor
    from openml.extensions.serialization import ModelSerializer


class OpenMLAPIConnector(ABC):
    """Base class for OpenML API connectors."""

    @abstractmethod
    def serializer(self) -> ModelSerializer:
        """Return the serializer for this API."""

    @abstractmethod
    def executor(self) -> ModelExecutor:
        """Return the executor for this API."""

    @classmethod
    @abstractmethod
    def supports(cls, model: Any) -> bool:
        """High-level check if this connector supports the model."""
