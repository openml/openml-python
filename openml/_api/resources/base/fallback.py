from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FallbackProxy:
    def __init__(self, *api_versions: Any):
        if not api_versions:
            raise ValueError("At least one API version must be provided")
        self._apis = api_versions

    def __getattr__(self, name: str) -> Any:
        api, attr = self._find_attr(name)
        if callable(attr):
            return self._wrap_callable(name, api, attr)
        return attr

    def _find_attr(self, name: str) -> tuple[Any, Any]:
        for api in self._apis:
            attr = getattr(api, name, None)
            if attr is not None:
                return api, attr
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def _wrap_callable(
        self,
        name: str,
        primary_api: Any,
        primary_attr: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return primary_attr(*args, **kwargs)
            except NotImplementedError:
                return self._call_fallbacks(name, primary_api, *args, **kwargs)

        return wrapper

    def _call_fallbacks(
        self,
        name: str,
        skip_api: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        for api in self._apis:
            if api is skip_api:
                continue
            attr = getattr(api, name, None)
            if callable(attr):
                try:
                    return attr(*args, **kwargs)
                except NotImplementedError:
                    continue
        raise NotImplementedError(f"Could not fallback to any API for method: {name}")
