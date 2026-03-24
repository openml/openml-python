from __future__ import annotations

from typing import TYPE_CHECKING

from openml._api.clients import HTTPClient, MinIOClient
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
from openml.enums import APIVersion, ResourceType

if TYPE_CHECKING:
    from openml._api.resources import ResourceAPI

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
    """
    Builder for constructing API backend instances with all resource-specific APIs.

    This class organizes resource-specific API objects (datasets, tasks,
    flows, evaluations, runs, setups, studies, etc.) and provides a
    centralized access point for both the primary API version and an
    optional fallback API version.

    The constructor automatically initializes:

    - HTTPClient for the primary API version
    - Optional HTTPClient for a fallback API version
    - MinIOClient for file storage operations
    - Resource-specific API instances, optionally wrapped with fallback proxies

    Parameters
    ----------
    api_version : APIVersion
        The primary API version to use for all resource APIs and HTTP communication.
    fallback_api_version : APIVersion | None, default=None
        Optional fallback API version to wrap resource APIs with a FallbackProxy.

    Attributes
    ----------
    dataset : ResourceAPI | FallbackProxy
        API interface for dataset resources.
    task : ResourceAPI | FallbackProxy
        API interface for task resources.
    evaluation_measure : ResourceAPI | FallbackProxy
        API interface for evaluation measure resources.
    estimation_procedure : ResourceAPI | FallbackProxy
        API interface for estimation procedure resources.
    evaluation : ResourceAPI | FallbackProxy
        API interface for evaluation resources.
    flow : ResourceAPI | FallbackProxy
        API interface for flow resources.
    study : ResourceAPI | FallbackProxy
        API interface for study resources.
    run : ResourceAPI | FallbackProxy
        API interface for run resources.
    setup : ResourceAPI | FallbackProxy
        API interface for setup resources.
    http_client : HTTPClient
        Client for HTTP communication using the primary API version.
    fallback_http_client : HTTPClient | None
        Client for HTTP communication using the fallback API version, if provided.
    minio_client : MinIOClient
        Client for file storage operations (MinIO/S3).
    """

    dataset: ResourceAPI | FallbackProxy
    task: ResourceAPI | FallbackProxy
    evaluation_measure: ResourceAPI | FallbackProxy
    estimation_procedure: ResourceAPI | FallbackProxy
    evaluation: ResourceAPI | FallbackProxy
    flow: ResourceAPI | FallbackProxy
    study: ResourceAPI | FallbackProxy
    run: ResourceAPI | FallbackProxy
    setup: ResourceAPI | FallbackProxy
    http_client: HTTPClient
    fallback_http_client: HTTPClient | None
    minio_client: MinIOClient

    def __init__(self, api_version: APIVersion, fallback_api_version: APIVersion | None = None):
        # initialize clients and resource APIs in-place
        self._build(api_version, fallback_api_version)

    def _build(self, api_version: APIVersion, fallback_api_version: APIVersion | None) -> None:
        """
        Construct an APIBackendBuilder instance from a configuration.

        This method initializes HTTP and MinIO clients, creates resource-specific
        API instances for the primary API version, and optionally wraps them
        with fallback proxies if a fallback API version is configured.

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
                resource_apis[resource_type] = resource_api_cls(
                    primary_http_client, minio_client
                )  # type: ignore[call-arg]
            else:
                resource_apis[resource_type] = resource_api_cls(primary_http_client)

        Parameters
        ----------
        api_version : APIVersion
            Primary API version to use for resource access.
        fallback_api_version : APIVersion | None
            Optional fallback API version for compatibility.

        Returns
        -------
        APIBackendBuilder
            Builder instance with all resource API interfaces initialized.
        """
        minio_client = MinIOClient()
        primary_http_client = HTTPClient(api_version=api_version)

        self.http_client = primary_http_client
        self.minio_client = minio_client
        self.fallback_http_client = None

        resource_apis: dict[ResourceType, ResourceAPI | FallbackProxy] = {}
        for resource_type, resource_api_cls in API_REGISTRY[api_version].items():
            if resource_type == ResourceType.DATASET:
                resource_apis[resource_type] = resource_api_cls(  # type: ignore[call-arg]
                    primary_http_client, minio_client
                )
            else:
                resource_apis[resource_type] = resource_api_cls(primary_http_client)

        if fallback_api_version is not None:
            fallback_http_client = HTTPClient(api_version=fallback_api_version)
            self.fallback_http_client = fallback_http_client

            fallback_resource_apis: dict[ResourceType, ResourceAPI] = {}
            for resource_type, resource_api_cls in API_REGISTRY[fallback_api_version].items():
                if resource_type == ResourceType.DATASET:
                    fallback_resource_apis[resource_type] = resource_api_cls(  # type: ignore[call-arg]
                        fallback_http_client, minio_client
                    )
                else:
                    fallback_resource_apis[resource_type] = resource_api_cls(fallback_http_client)
            resource_apis = {
                name: FallbackProxy(resource_apis[name], fallback_resource_apis[name])
                for name in resource_apis
            }

        self.dataset = resource_apis[ResourceType.DATASET]
        self.task = resource_apis[ResourceType.TASK]
        self.evaluation_measure = resource_apis[ResourceType.EVALUATION_MEASURE]
        self.estimation_procedure = resource_apis[ResourceType.ESTIMATION_PROCEDURE]
        self.evaluation = resource_apis[ResourceType.EVALUATION]
        self.flow = resource_apis[ResourceType.FLOW]
        self.study = resource_apis[ResourceType.STUDY]
        self.run = resource_apis[ResourceType.RUN]
        self.setup = resource_apis[ResourceType.SETUP]
