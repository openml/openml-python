from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from openml._api.clients import HTTPCache, HTTPClient, MinIOClient
from openml._api.config import APIVersion, ResourceType
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
    from openml._api.resources.base import ResourceAPI

    from .config import Config

# Registry mapping API versions and resource types to their implementations
API_REGISTRY: dict[APIVersion, dict[ResourceType, type[ResourceAPI]]] = {
    APIVersion.V1: {
        ResourceType.DATASET: DatasetV1API,
        ResourceType.TASK: TaskV1API,
        ResourceType.EVALUATION_MEASURE: EvaluationMeasureV1API,
        ResourceType.ESTIMATION_PROCEDURE: EstimationProcedureV1API,
        ResourceType.EVALUATION: EvaluationV1API,
        ResourceType.FLOW: FlowV1API,
        ResourceType.STUDY: StudyV1API,
        ResourceType.RUN: RunV1API,
        ResourceType.SETUP: SetupV1API,
    },
    APIVersion.V2: {
        ResourceType.DATASET: DatasetV2API,
        ResourceType.TASK: TaskV2API,
        ResourceType.EVALUATION_MEASURE: EvaluationMeasureV2API,
        ResourceType.ESTIMATION_PROCEDURE: EstimationProcedureV2API,
        ResourceType.EVALUATION: EvaluationV2API,
        ResourceType.FLOW: FlowV2API,
        ResourceType.STUDY: StudyV2API,
        ResourceType.RUN: RunV2API,
        ResourceType.SETUP: SetupV2API,
    },
}


class APIBackendBuilder:
    """Builder for creating API backend with resource APIs."""

    def __init__(
        self,
        resource_apis: Mapping[ResourceType, ResourceAPI | FallbackProxy],
    ):
        self.dataset = resource_apis[ResourceType.DATASET]
        self.task = resource_apis[ResourceType.TASK]
        self.evaluation_measure = resource_apis[ResourceType.EVALUATION_MEASURE]
        self.estimation_procedure = resource_apis[ResourceType.ESTIMATION_PROCEDURE]
        self.evaluation = resource_apis[ResourceType.EVALUATION]
        self.flow = resource_apis[ResourceType.FLOW]
        self.study = resource_apis[ResourceType.STUDY]
        self.run = resource_apis[ResourceType.RUN]
        self.setup = resource_apis[ResourceType.SETUP]

    @classmethod
    def build(cls, config: Config) -> APIBackendBuilder:
        """Build API backend from configuration."""
        cache_dir = Path(config.cache.dir).expanduser()

        http_cache = HTTPCache(path=cache_dir, ttl=config.cache.ttl)
        minio_client = MinIOClient(path=cache_dir)

        primary_api_config = config.api_configs[config.api_version]
        primary_http_client = HTTPClient(
            server=primary_api_config.server,
            base_url=primary_api_config.base_url,
            api_key=primary_api_config.api_key,
            retries=config.connection.retries,
            retry_policy=config.connection.retry_policy,
            cache=http_cache,
            timeout=10,
        )

        resource_apis: dict[ResourceType, ResourceAPI] = {}
        for resource_type, resource_api_cls in API_REGISTRY[config.api_version].items():
            if resource_type == ResourceType.DATASET:
                resource_apis[resource_type] = resource_api_cls(primary_http_client, minio_client)  # type: ignore[call-arg]
            else:
                resource_apis[resource_type] = resource_api_cls(primary_http_client)

        if config.fallback_api_version is None:
            return cls(resource_apis)

        fallback_api_config = config.api_configs[config.fallback_api_version]
        fallback_http_client = HTTPClient(
            server=fallback_api_config.server,
            base_url=fallback_api_config.base_url,
            api_key=fallback_api_config.api_key,
            retries=config.connection.retries,
            retry_policy=config.connection.retry_policy,
            cache=http_cache,
            timeout=10,
        )

        fallback_resource_apis: dict[ResourceType, ResourceAPI] = {}
        for resource_type, resource_api_cls in API_REGISTRY[config.fallback_api_version].items():
            if resource_type == ResourceType.DATASET:
                fallback_resource_apis[resource_type] = resource_api_cls(
                    fallback_http_client,
                    minio_client,  # type: ignore[call-arg]
                )
            else:
                fallback_resource_apis[resource_type] = resource_api_cls(fallback_http_client)

        merged: dict[ResourceType, FallbackProxy] = {
            name: FallbackProxy(resource_apis[name], fallback_resource_apis[name])
            for name in resource_apis
        }

        return cls(merged)
