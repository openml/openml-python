from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from openml._api.clients import HTTPClient, MinIOClient
from openml._api.resources import API_REGISTRY, FallbackProxy
from openml.enums import ResourceType

if TYPE_CHECKING:
    from openml._api.resources import ResourceAPI
    from openml.enums import APIVersion


class APIBackendBuilder:
    """
    Builder class for constructing API backend instances.

    This class organizes resource-specific API objects (datasets, tasks,
    flows, evaluations, runs, setups, studies, etc.) and provides a
    centralized access point for both primary and optional fallback APIs.

    Parameters
    ----------
    resource_apis : Mapping[ResourceType, ResourceAPI | FallbackProxy]
        Mapping of resource types to their corresponding API instances
        or fallback proxies.

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
        Client for HTTP Communication.
    fallback_http_client : HTTPClient | None
        Fallback Client for HTTP Communication.
    minio_client : MinIOClient
        Client for MinIO Communication.
    """

    def __init__(
        self,
        clients: Mapping[str, HTTPClient | MinIOClient | None],
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
        self.http_client = clients["http_client"]
        self.fallback_http_client = clients["fallback_http_client"]
        self.minio_client = clients["minio_client"]

    @classmethod
    def build(
        cls,
        api_version: APIVersion,
        fallback_api_version: APIVersion | None,
    ) -> APIBackendBuilder:
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
        clients: dict[str, HTTPClient | MinIOClient | None] = {
            "http_client": primary_http_client,
            "fallback_http_client": None,
            "minio_client": minio_client,
        }

        resource_apis: dict[ResourceType, ResourceAPI] = {}
        for resource_type, resource_api_cls in API_REGISTRY[api_version].items():
            resource_apis[resource_type] = resource_api_cls(primary_http_client, minio_client)

        if fallback_api_version is None:
            return cls(clients, resource_apis)

        fallback_http_client = HTTPClient(api_version=fallback_api_version)
        clients["fallback_http_client"] = fallback_http_client

        fallback_resource_apis: dict[ResourceType, ResourceAPI] = {}
        for resource_type, resource_api_cls in API_REGISTRY[fallback_api_version].items():
            fallback_resource_apis[resource_type] = resource_api_cls(
                fallback_http_client, minio_client
            )

        merged: dict[ResourceType, FallbackProxy] = {
            name: FallbackProxy(resource_apis[name], fallback_resource_apis[name])
            for name in resource_apis
        }

        return cls(clients, merged)
