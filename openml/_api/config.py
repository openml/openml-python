from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class APIVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


class ResourceType(str, Enum):
    DATASET = "dataset"
    TASK = "task"
    TASK_TYPE = "task_type"
    EVALUATION_MEASURE = "evaluation_measure"
    ESTIMATION_PROCEDURE = "estimation_procedure"
    EVALUATION = "evaluation"
    FLOW = "flow"
    STUDY = "study"
    RUN = "run"
    SETUP = "setup"
    USER = "user"


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
class ConnectionConfig:
    retries: int = 3
    retry_policy: RetryPolicy = RetryPolicy.HUMAN


@dataclass
class CacheConfig:
    dir: str = "~/.openml/cache"
    ttl: int = 60 * 60 * 24 * 7  # one week


class Settings:
    """Settings container that reads from openml.config on access."""

    _instance: Settings | None = None

    def __init__(self) -> None:
        self.api_configs: dict[str, APIConfig] = {}
        self.connection = ConnectionConfig()
        self.cache = CacheConfig()
        self._initialized = False

    @classmethod
    def get(cls) -> Settings:
        """Get settings singleton, creating on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the settings singleton. Useful for testing."""
        cls._instance = None

    def get_api_config(self, version: str) -> APIConfig:
        """Get API config for a version, with lazy initialization from openml.config."""
        if not self._initialized:
            self._init_from_legacy_config()
        if version not in self.api_configs:
            raise NotImplementedError(
                f"API {version} is not yet available. "
                f"Supported versions: {list(self.api_configs.keys())}"
            )
        return self.api_configs[version]

    def _init_from_legacy_config(self) -> None:
        """Lazy init from openml.config to avoid circular imports."""
        if self._initialized:
            return

        # Import here (not at module level) to avoid circular imports.
        # We read from openml.config to integrate with the existing config system
        # where users set their API key, server, cache directory, etc.
        # This avoids duplicating those settings with hardcoded values.
        import openml.config as legacy

        server_url = legacy.server
        server_base = server_url.rsplit("/api", 1)[0] + "/" if "/api" in server_url else server_url

        self.api_configs["v1"] = APIConfig(
            server=server_base,
            base_url="api/v1/xml/",
            api_key=legacy.apikey,
        )

        # Sync connection- and cache- settings from legacy config
        self.connection = ConnectionConfig(
            retries=legacy.connection_n_retries,
            retry_policy=RetryPolicy(legacy.retry_policy),
        )
        self.cache = CacheConfig(
            dir=str(legacy._root_cache_directory),
        )

        self._initialized = True
