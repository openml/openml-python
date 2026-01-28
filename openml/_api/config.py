from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RetryPolicy(str, Enum):
    HUMAN = "human"
    ROBOT = "robot"


@dataclass
class APIConfig:
    server: str
    base_url: str
    api_key: str
    timeout: int = 10  # seconds


@dataclass
class APISettings:
    v1: APIConfig
    v2: APIConfig


@dataclass
class ConnectionConfig:
    retries: int = 3
    retry_policy: RetryPolicy = RetryPolicy.HUMAN


@dataclass
class CacheConfig:
    dir: str = "~/.openml/cache"
    ttl: int = 60 * 60 * 24 * 7  # one week


@dataclass
class Settings:
    api: APISettings
    connection: ConnectionConfig
    cache: CacheConfig


settings = Settings(
    api=APISettings(
        v1=APIConfig(
            server="https://www.openml.org/",
            base_url="api/v1/xml/",
            api_key="...",
        ),
        v2=APIConfig(
            server="http://127.0.0.1:8001/",
            base_url="",
            api_key="...",
        ),
    ),
    connection=ConnectionConfig(),
    cache=CacheConfig(),
)
