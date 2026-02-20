from __future__ import annotations

from dataclasses import dataclass, field

from openml.enums import APIVersion, RetryPolicy

from ._utils import _resolve_default_cache_dir


@dataclass
class APIConfig:
    """
    Configuration for a specific OpenML API version.

    Parameters
    ----------
    server : str
        Base server URL for the API.
    base_url : str
        API-specific base path appended to the server URL.
    api_key : str | None, default=None
        API key used for authentication. If None, requests are made
        without authentication.
    """

    server: str
    base_url: str
    api_key: str | None = None


@dataclass
class ConnectionConfig:
    """
    Configuration for HTTP connection behavior.

    Parameters
    ----------
    retries : int
        Number of retry attempts for failed requests.
    retry_policy : RetryPolicy
        Policy for determining delays between retries (human-like or robot-like).
    """

    retries: int
    retry_policy: RetryPolicy


@dataclass
class Config:
    """
    Global configuration for the OpenML Python client.

    Includes API versions, connection settings, and caching options.

    Attributes
    ----------
    api_version : APIVersion
        Primary API version to use (default is V1).
    fallback_api_version : APIVersion or None
        Optional fallback API version if the primary API does not support certain operations.
    cache_dir : str
        Path to the directory where cached files will be stored.
    api_configs : dict of APIVersion to APIConfig
        Mapping from API version to its server/base URL and API key configuration.
    connection : ConnectionConfig
        Settings for request retries and retry policy.
    """

    api_version: APIVersion = APIVersion.V1
    fallback_api_version: APIVersion | None = None
    cache_dir: str = str(_resolve_default_cache_dir())

    api_configs: dict[APIVersion, APIConfig] = field(
        default_factory=lambda: {
            APIVersion.V1: APIConfig(
                server="https://www.openml.org/",
                base_url="api/v1/xml/",
                api_key=None,
            ),
            APIVersion.V2: APIConfig(
                server="http://localhost:8002/",
                base_url="",
                api_key=None,
            ),
        }
    )

    connection: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(
            retries=5,
            retry_policy=RetryPolicy.HUMAN,
        )
    )
