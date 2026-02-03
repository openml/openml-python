from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from .builder import APIBackendBuilder
from .config import Config

if TYPE_CHECKING:
    from openml._api.resources import (
        DatasetAPI,
        EstimationProcedureAPI,
        EvaluationAPI,
        EvaluationMeasureAPI,
        FlowAPI,
        RunAPI,
        SetupAPI,
        StudyAPI,
        TaskAPI,
    )


class APIBackend:
    _instance: APIBackend | None = None

    def __init__(self, config: Config | None = None):
        self._config: Config = config or Config()
        self._backend = APIBackendBuilder.build(self._config)

    @property
    def dataset(self) -> DatasetAPI:
        return cast("DatasetAPI", self._backend.dataset)

    @property
    def task(self) -> TaskAPI:
        return cast("TaskAPI", self._backend.task)

    @property
    def evaluation_measure(self) -> EvaluationMeasureAPI:
        return cast("EvaluationMeasureAPI", self._backend.evaluation_measure)

    @property
    def estimation_procedure(self) -> EstimationProcedureAPI:
        return cast("EstimationProcedureAPI", self._backend.estimation_procedure)

    @property
    def evaluation(self) -> EvaluationAPI:
        return cast("EvaluationAPI", self._backend.evaluation)

    @property
    def flow(self) -> FlowAPI:
        return cast("FlowAPI", self._backend.flow)

    @property
    def study(self) -> StudyAPI:
        return cast("StudyAPI", self._backend.study)

    @property
    def run(self) -> RunAPI:
        return cast("RunAPI", self._backend.run)

    @property
    def setup(self) -> SetupAPI:
        return cast("SetupAPI", self._backend.setup)

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
    def get_config_value(cls, key: str) -> Any:
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

    @classmethod
    def get_config_values(cls, keys: list[str]) -> list[Any]:
        values = []
        for key in keys:
            value = cls.get_config_value(key)
            values.append(value)
        return values

    @classmethod
    def set_config_values(cls, config_dict: dict[str, Any]) -> None:
        config = cls.get_instance()._config

        for key, value in config_dict.items():
            keys = key.split(".")
            parent = config
            for k in keys[:-1]:
                parent = parent[k] if isinstance(parent, dict) else getattr(parent, k)
            if isinstance(parent, dict):
                parent[keys[-1]] = value
            else:
                setattr(parent, keys[-1], value)

        cls.set_config(config)
