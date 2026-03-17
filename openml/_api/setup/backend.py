from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import openml

from .builder import APIBackendBuilder

if TYPE_CHECKING:
    from openml._api.clients import HTTPClient, MinIOClient
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
    """
    Central backend for accessing all OpenML API resource interfaces.

    This class provides a singleton interface to dataset, task, flow,
    evaluation, run, setup, study, and other resource APIs. It also
    manages configuration through a nested ``Config`` object and
    allows dynamic retrieval and updating of configuration values.

    Parameters
    ----------
    config : Config, optional
        Optional configuration object. If not provided, a default
        ``Config`` instance is created.

    Attributes
    ----------
    dataset : DatasetAPI
        Interface for dataset-related API operations.
    task : TaskAPI
        Interface for task-related API operations.
    evaluation_measure : EvaluationMeasureAPI
        Interface for evaluation measure-related API operations.
    estimation_procedure : EstimationProcedureAPI
        Interface for estimation procedure-related API operations.
    evaluation : EvaluationAPI
        Interface for evaluation-related API operations.
    flow : FlowAPI
        Interface for flow-related API operations.
    study : StudyAPI
        Interface for study-related API operations.
    run : RunAPI
        Interface for run-related API operations.
    setup : SetupAPI
        Interface for setup-related API operations.
    """

    _instance: ClassVar[APIBackend | None] = None
    _backends: ClassVar[dict[str, APIBackendBuilder]] = {}

    @property
    def _backend(self) -> APIBackendBuilder:
        api_version = openml.config.api_version
        fallback_api_version = openml.config.fallback_api_version
        key = f"{api_version}_{fallback_api_version}"

        if key not in self._backends:
            _backend = APIBackendBuilder(
                api_version=api_version,
                fallback_api_version=fallback_api_version,
            )
            self._backends[key] = _backend

        return self._backends[key]

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

    @property
    def http_client(self) -> HTTPClient:
        return cast("HTTPClient", self._backend.http_client)

    @property
    def fallback_http_client(self) -> HTTPClient | None:
        return cast("HTTPClient | None", self._backend.fallback_http_client)

    @property
    def minio_client(self) -> MinIOClient:
        return cast("MinIOClient", self._backend.minio_client)

    @classmethod
    def get_instance(cls) -> APIBackend:
        """
        Get the singleton instance of the APIBackend.

        Returns
        -------
        APIBackend
            Singleton instance of the backend.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
