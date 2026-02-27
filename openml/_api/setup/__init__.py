"""API backend setup module."""

from .backend import APIBackend
from .builder import API_REGISTRY, APIBackendBuilder

_backend = APIBackend.get_instance()

__all__ = [
    "API_REGISTRY",
    "APIBackend",
    "APIBackendBuilder",
    "_backend",
]
