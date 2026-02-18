from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from openml.exceptions import OpenMLNotSupportedError

if TYPE_CHECKING:
    from .base import ResourceAPI


class FallbackProxy:
    """
    Proxy object that provides transparent fallback between two API versions.

    Parameters
    ----------
    primary_api : Any
        Primary API implementation.
    fallback_api : Any
        Secondary API implementation used if the primary raises
        ``OpenMLNotSupportedError``.
    """

    def __init__(self, primary_api: ResourceAPI, fallback_api: ResourceAPI):
        self._primary = primary_api
        self._fallback = fallback_api

    def __getattr__(self, name: str) -> Any:
        primary_attr = getattr(self._primary, name, None)
        fallback_attr = getattr(self._fallback, name, None)

        if primary_attr is None and fallback_attr is None:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

        # If attribute exists on primary
        if primary_attr is not None:
            if callable(primary_attr):
                return self._wrap_callable(name, primary_attr)
            return primary_attr

        # Otherwise return fallback attribute directly
        return fallback_attr

    def _wrap_callable(
        self,
        name: str,
        primary_attr: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return primary_attr(*args, **kwargs)
            except OpenMLNotSupportedError:
                fallback_attr = getattr(self._fallback, name, None)
                if callable(fallback_attr):
                    return fallback_attr(*args, **kwargs)
                raise OpenMLNotSupportedError(
                    f"Method '{name}' not supported by primary or fallback API"
                ) from None

        return wrapper
