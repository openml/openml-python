from .backend import APIBackend
from .builder import APIBackendBuilder

_backend = APIBackend.get_instance()

__all__ = [
    "APIBackend",
    "APIBackendBuilder",
    "_backend",
]
