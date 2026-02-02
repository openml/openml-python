from __future__ import annotations

from dataclasses import dataclass, field

from openml.enums import APIVersion, RetryPolicy

from ._utils import _resolve_default_cache_dir


@dataclass
class APIConfig:
    server: str
    base_url: str
    api_key: str


@dataclass
class ConnectionConfig:
    retries: int
    retry_policy: RetryPolicy
    timeout: int


@dataclass
class CacheConfig:
    dir: str
    ttl: int


@dataclass
class Config:
    api_version: APIVersion = APIVersion.V1
    fallback_api_version: APIVersion | None = None

    api_configs: dict[APIVersion, APIConfig] = field(
        default_factory=lambda: {
            APIVersion.V1: APIConfig(
                server="https://www.openml.org/",
                base_url="api/v1/xml/",
                api_key="",
            ),
            APIVersion.V2: APIConfig(
                server="http://localhost:8002/",
                base_url="",
                api_key="",
            ),
        }
    )

    connection: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(
            retries=5,
            retry_policy=RetryPolicy.HUMAN,
            timeout=10,
        )
    )

    cache: CacheConfig = field(
        default_factory=lambda: CacheConfig(
            dir=str(_resolve_default_cache_dir()),
            ttl=60 * 60 * 24 * 7,
        )
    )
