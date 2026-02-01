"""OpenML API module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openml._api.runtime.instance import _backend as backend

if TYPE_CHECKING:
    from openml._api.runtime.core import APIBackend

__all__ = ["api_context"]


class APIContext:
    """API context for accessing the OpenML backend."""

    @property
    def backend(self) -> APIBackend:
        """Get the API backend instance."""
        return backend


api_context = APIContext()
