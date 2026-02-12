from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from openml._api.clients import HTTPCache, HTTPClient, MinIOClient
from openml._api.resources import API_REGISTRY, FallbackProxy, ResourceAPI
from openml.enums import ResourceType

if TYPE_CHECKING:
    from .config import Config


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
    """

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
        )

        resource_apis: dict[ResourceType, ResourceAPI] = {}
        for resource_type, resource_api_cls in API_REGISTRY[config.api_version].items():
            resource_apis[resource_type] = resource_api_cls(primary_http_client, minio_client)

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
        )

        fallback_resource_apis: dict[ResourceType, ResourceAPI] = {}
        for resource_type, resource_api_cls in API_REGISTRY[config.fallback_api_version].items():
            fallback_resource_apis[resource_type] = resource_api_cls(
                fallback_http_client, minio_client
            )

        merged: dict[ResourceType, FallbackProxy] = {
            name: FallbackProxy(resource_apis[name], fallback_resource_apis[name])
            for name in resource_apis
        }

        return cls(merged)
