from __future__ import annotations

from typing import TYPE_CHECKING

from openml._api.clients import HTTPClient, MinIOClient
from openml._api.resources import (
    API_REGISTRY,
    FallbackProxy,
)
from openml.enums import ResourceType

if TYPE_CHECKING:
    from openml._api.resources import ResourceAPI
    from openml.enums import APIVersion


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

        Parameters
        ----------
        config : Config
            Configuration object containing API versions, endpoints, cache
            settings, and connection parameters.

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
            resource_apis[resource_type] = resource_api_cls(primary_http_client, minio_client)

        if fallback_api_version is not None:
            fallback_http_client = HTTPClient(api_version=fallback_api_version)
            self.fallback_http_client = fallback_http_client

            fallback_resource_apis: dict[ResourceType, ResourceAPI | FallbackProxy] = {}
            for resource_type, resource_api_cls in API_REGISTRY[fallback_api_version].items():
                fallback_resource_apis[resource_type] = resource_api_cls(
                    fallback_http_client, minio_client
                )

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
