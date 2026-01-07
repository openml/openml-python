from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DelayMethod = Literal["human", "robot"]


@dataclass
class APIConfig:
    server: str
    base_url: str
    key: str


@dataclass
class APISettings:
    v1: APIConfig
    v2: APIConfig


@dataclass
class ConnectionConfig:
    retries: int = 3
    delay_method: DelayMethod = "human"
    delay_time: int = 1  # seconds

    def __post_init__(self) -> None:
        if self.delay_method not in ("human", "robot"):
            raise ValueError(f"delay_method must be 'human' or 'robot', got {self.delay_method}")


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
            key="...",
        ),
        v2=APIConfig(
            server="http://127.0.0.1:8001/",
            base_url="",
            key="...",
        ),
    ),
    connection=ConnectionConfig(),
    cache=CacheConfig(),
)
