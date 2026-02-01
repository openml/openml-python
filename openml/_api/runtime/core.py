from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openml._api.clients import HTTPCache, HTTPClient, MinIOClient
from openml._api.config import Settings
from openml._api.resources import (
    DatasetV1API,
    DatasetV2API,
    EstimationProcedureV1API,
    EstimationProcedureV2API,
    EvaluationMeasureV1API,
    EvaluationMeasureV2API,
    EvaluationV1API,
    EvaluationV2API,
    FallbackProxy,
    FlowV1API,
    FlowV2API,
    RunV1API,
    RunV2API,
    SetupV1API,
    SetupV2API,
    StudyV1API,
    StudyV2API,
    TaskV1API,
    TaskV2API,
)

if TYPE_CHECKING:
    from openml._api.resources.base import (
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
    def __init__(  # noqa: PLR0913
        self,
        *,
        dataset: DatasetAPI | FallbackProxy,
        task: TaskAPI | FallbackProxy,
        evaluation_measure: EvaluationMeasureAPI | FallbackProxy,
        estimation_procedure: EstimationProcedureAPI | FallbackProxy,
        evaluation: EvaluationAPI | FallbackProxy,
        flow: FlowAPI | FallbackProxy,
        study: StudyAPI | FallbackProxy,
        run: RunAPI | FallbackProxy,
        setup: SetupAPI | FallbackProxy,
    ):
        self.dataset = dataset
        self.task = task
        self.evaluation_measure = evaluation_measure
        self.estimation_procedure = estimation_procedure
        self.evaluation = evaluation
        self.flow = flow
        self.study = study
        self.run = run
        self.setup = setup

    @classmethod
    def build(cls, version: str, *, strict: bool) -> APIBackend:
        settings = Settings.get()

        # Get config for v1. On first access, this triggers lazy initialization
        # from openml.config, reading the user's actual API key, server URL,
        # cache directory, and retry settings. This avoids circular imports
        # (openml.config is imported inside the method, not at module load time)
        # and ensures we use the user's configured values rather than hardcoded defaults.
        v1_config = settings.get_api_config("v1")

        http_cache = HTTPCache(
            path=Path(settings.cache.dir).expanduser(),
            ttl=settings.cache.ttl,
        )
        minio_client = MinIOClient(
            path=Path(settings.cache.dir).expanduser(),
        )

        v1_http_client = HTTPClient(
            server=v1_config.server,
            base_url=v1_config.base_url,
            api_key=v1_config.api_key,
            timeout=v1_config.timeout,
            retries=settings.connection.retries,
            retry_policy=settings.connection.retry_policy,
            cache=http_cache,
        )
        v1_dataset = DatasetV1API(v1_http_client, minio_client)
        v1_task = TaskV1API(v1_http_client)
        v1_evaluation_measure = EvaluationMeasureV1API(v1_http_client)
        v1_estimation_procedure = EstimationProcedureV1API(v1_http_client)
        v1_evaluation = EvaluationV1API(v1_http_client)
        v1_flow = FlowV1API(v1_http_client)
        v1_study = StudyV1API(v1_http_client)
        v1_run = RunV1API(v1_http_client)
        v1_setup = SetupV1API(v1_http_client)

        v1 = cls(
            dataset=v1_dataset,
            task=v1_task,
            evaluation_measure=v1_evaluation_measure,
            estimation_procedure=v1_estimation_procedure,
            evaluation=v1_evaluation,
            flow=v1_flow,
            study=v1_study,
            run=v1_run,
            setup=v1_setup,
        )

        if version == "v1":
            return v1

        # V2 support. Currently v2 is not yet available,
        # so get_api_config("v2") raises NotImplementedError. When v2 becomes available,
        # its config will be added to Settings._init_from_legacy_config().
        # In strict mode: propagate the error.
        # In non-strict mode: silently fall back to v1 only.
        try:
            v2_config = settings.get_api_config("v2")
        except NotImplementedError:
            if strict:
                raise
            # Non-strict mode: fall back to v1 only
            return v1

        v2_http_client = HTTPClient(
            server=v2_config.server,
            base_url=v2_config.base_url,
            api_key=v2_config.api_key,
            timeout=v2_config.timeout,
            retries=settings.connection.retries,
            retry_policy=settings.connection.retry_policy,
            cache=http_cache,
        )
        v2_dataset = DatasetV2API(v2_http_client, minio_client)
        v2_task = TaskV2API(v2_http_client)
        v2_evaluation_measure = EvaluationMeasureV2API(v2_http_client)
        v2_estimation_procedure = EstimationProcedureV2API(v2_http_client)
        v2_evaluation = EvaluationV2API(v2_http_client)
        v2_flow = FlowV2API(v2_http_client)
        v2_study = StudyV2API(v2_http_client)
        v2_run = RunV2API(v2_http_client)
        v2_setup = SetupV2API(v2_http_client)

        v2 = cls(
            dataset=v2_dataset,
            task=v2_task,
            evaluation_measure=v2_evaluation_measure,
            estimation_procedure=v2_estimation_procedure,
            evaluation=v2_evaluation,
            flow=v2_flow,
            study=v2_study,
            run=v2_run,
            setup=v2_setup,
        )

        if strict:
            return v2

        fallback_dataset = FallbackProxy(v1_dataset, v2_dataset)
        fallback_task = FallbackProxy(v1_task, v2_task)
        fallback_evaluation_measure = FallbackProxy(v1_evaluation_measure, v2_evaluation_measure)
        fallback_estimation_procedure = FallbackProxy(
            v1_estimation_procedure, v2_estimation_procedure
        )
        fallback_evaluation = FallbackProxy(v1_evaluation, v2_evaluation)
        fallback_flow = FallbackProxy(v1_flow, v2_flow)
        fallback_study = FallbackProxy(v1_study, v2_study)
        fallback_run = FallbackProxy(v1_run, v2_run)
        fallback_setup = FallbackProxy(v1_setup, v2_setup)

        return cls(
            dataset=fallback_dataset,
            task=fallback_task,
            evaluation_measure=fallback_evaluation_measure,
            estimation_procedure=fallback_estimation_procedure,
            evaluation=fallback_evaluation,
            flow=fallback_flow,
            study=fallback_study,
            run=fallback_run,
            setup=fallback_setup,
        )
