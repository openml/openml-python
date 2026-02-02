from __future__ import annotations

from copy import deepcopy
from typing import Any

from .builder import APIBackendBuilder
from .config import Config


class APIBackend:
    _instance: APIBackend | None = None

    def __init__(self, config: Config | None = None):
        self._config: Config = config or Config()
        self._backend = APIBackendBuilder.build(self._config)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying backend.
        Called only if attribute is not found on RuntimeBackend.
        """
        return getattr(self._backend, name)

    @classmethod
    def get_instance(cls) -> APIBackend:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_config(cls) -> Config:
        return deepcopy(cls.get_instance()._config)

    @classmethod
    def set_config(cls, config: Config) -> None:
        instance = cls.get_instance()
        instance._config = config
        instance._backend = APIBackendBuilder.build(config)

    @classmethod
    def get_config_value(cls, key: str) -> Config:
        keys = key.split(".")
        config_value = cls.get_instance()._config
        for k in keys:
            if isinstance(config_value, dict):
                config_value = config_value[k]
            else:
                config_value = getattr(config_value, k)
        return deepcopy(config_value)

    @classmethod
    def set_config_value(cls, key: str, value: Any) -> None:
        keys = key.split(".")
        config = cls.get_instance()._config
        parent = config
        for k in keys[:-1]:
            parent = parent[k] if isinstance(parent, dict) else getattr(parent, k)
        if isinstance(parent, dict):
            parent[keys[-1]] = value
        else:
            setattr(parent, keys[-1], value)
        cls.set_config(config)
