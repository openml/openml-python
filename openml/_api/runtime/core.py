from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openml._api.clients import HTTPCache, HTTPClient, MinIOClient
from openml._api.config import settings
from openml._api.resources import (
    DatasetsV1,
    DatasetsV2,
    TasksV1,
    TasksV2,
)

if TYPE_CHECKING:
    from openml._api.resources.base import DatasetsAPI, TasksAPI


class APIBackend:
    def __init__(self, *, datasets: DatasetsAPI, tasks: TasksAPI):
        self.datasets = datasets
        self.tasks = tasks


def build_backend(version: str, *, strict: bool) -> APIBackend:
    http_cache = HTTPCache(
        path=Path(settings.cache.dir),
        ttl=settings.cache.ttl,
    )
    minio_client = MinIOClient(path=Path(settings.cache.dir))
    v1_http_client = HTTPClient(
        server=settings.api.v1.server,
        base_url=settings.api.v1.base_url,
        api_key=settings.api.v1.api_key,
        timeout=settings.api.v1.timeout,
        retries=settings.connection.retries,
        retry_policy=settings.connection.retry_policy,
        cache=http_cache,
    )
    v2_http_client = HTTPClient(
        server=settings.api.v2.server,
        base_url=settings.api.v2.base_url,
        api_key=settings.api.v2.api_key,
        timeout=settings.api.v2.timeout,
        retries=settings.connection.retries,
        retry_policy=settings.connection.retry_policy,
        cache=http_cache,
    )

    v1 = APIBackend(
        datasets=DatasetsV1(v1_http_client, minio_client),
        tasks=TasksV1(v1_http_client),
    )

    if version == "v1":
        return v1

    v2 = APIBackend(
        datasets=DatasetsV2(v2_http_client, minio_client),
        tasks=TasksV2(v2_http_client),
    )

    if strict:
        return v2

    return v1


class APIContext:
    def __init__(self) -> None:
        self._backend = build_backend("v1", strict=False)

    def set_version(self, version: str, *, strict: bool = False) -> None:
        self._backend = build_backend(version=version, strict=strict)

    @property
    def backend(self) -> APIBackend:
        return self._backend
