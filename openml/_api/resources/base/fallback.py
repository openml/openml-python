from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openml.exceptions import OpenMLNotSupportedError

if TYPE_CHECKING:
    from collections.abc import Callable


class FallbackProxy:
    """
    Proxy object that provides transparent fallback across multiple API versions.

    This class delegates attribute access to a sequence of API implementations.
    When a callable attribute is invoked and raises ``OpenMLNotSupportedError``,
    the proxy automatically attempts the same method on subsequent API instances
    until one succeeds.

    Parameters
    ----------
    *api_versions : Any
        One or more API implementation instances ordered by priority.
        The first API is treated as the primary implementation, and
        subsequent APIs are used as fallbacks.

    Raises
    ------
    ValueError
        If no API implementations are provided.

    Notes
    -----
    Attribute lookup is performed dynamically via ``__getattr__``.
    Only methods that raise ``OpenMLNotSupportedError`` trigger fallback
    behavior. Other exceptions are propagated immediately.
    """

    def __init__(self, *api_versions: Any):
        if not api_versions:
            raise ValueError("At least one API version must be provided")
        self._apis = api_versions

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically resolve attribute access across API implementations.

        Parameters
        ----------
        name : str
            Name of the attribute being accessed.

        Returns
        -------
        Any
            The resolved attribute. If it is callable, a wrapped function
            providing fallback behavior is returned.

        Raises
        ------
        AttributeError
            If none of the API implementations define the attribute.
        """
        api, attr = self._find_attr(name)
        if callable(attr):
            return self._wrap_callable(name, api, attr)
        return attr

    def _find_attr(self, name: str) -> tuple[Any, Any]:
        """
        Find the first API implementation that defines a given attribute.

        Parameters
        ----------
        name : str
            Name of the attribute to search for.

        Returns
        -------
        tuple of (Any, Any)
            The API instance and the corresponding attribute.

        Raises
        ------
        AttributeError
            If no API implementation defines the attribute.
        """
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
        """
        Wrap a callable attribute to enable fallback behavior.

        Parameters
        ----------
        name : str
            Name of the method being wrapped.
        primary_api : Any
            Primary API instance providing the callable.
        primary_attr : Callable[..., Any]
            Callable attribute obtained from the primary API.

        Returns
        -------
        Callable[..., Any]
            Wrapped function that attempts the primary call first and
            falls back to other APIs if ``OpenMLNotSupportedError`` is raised.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return primary_attr(*args, **kwargs)
            except OpenMLNotSupportedError:
                return self._call_fallbacks(name, primary_api, *args, **kwargs)

        return wrapper

    def _call_fallbacks(
        self,
        name: str,
        skip_api: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Attempt to call a method on fallback API implementations.

        Parameters
        ----------
        name : str
            Name of the method to invoke.
        skip_api : Any
            API instance to skip (typically the primary API that already failed).
        *args : Any
            Positional arguments passed to the method.
        **kwargs : Any
            Keyword arguments passed to the method.

        Returns
        -------
        Any
            Result returned by the first successful fallback invocation.

        Raises
        ------
        OpenMLNotSupportedError
            If all API implementations either do not define the method
            or raise ``OpenMLNotSupportedError``.
        """
        for api in self._apis:
            if api is skip_api:
                continue
            attr = getattr(api, name, None)
            if callable(attr):
                try:
                    return attr(*args, **kwargs)
                except OpenMLNotSupportedError:
                    continue
        raise OpenMLNotSupportedError(f"Could not fallback to any API for method: {name}")
