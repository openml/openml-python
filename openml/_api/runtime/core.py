from __future__ import annotations

from typing import TYPE_CHECKING

from openml._api.config import settings
from openml._api.http.client import HTTPClient
from openml._api.resources import (
    DatasetsV1,
    DatasetsV2,
    FlowsV1,
    FlowsV2,
    TasksV1,
    TasksV2,
)

if TYPE_CHECKING:
    from openml._api.resources.base import DatasetsAPI, FlowsAPI, TasksAPI


class APIBackend:
    def __init__(self, *, datasets: DatasetsAPI, tasks: TasksAPI, flows: FlowsAPI):
        self.datasets = datasets
        self.tasks = tasks
        self.flows = flows


def build_backend(version: str, *, strict: bool) -> APIBackend:
    v1_http = HTTPClient(config=settings.api.v1)
    v2_http = HTTPClient(config=settings.api.v2)

    v1 = APIBackend(
        datasets=DatasetsV1(v1_http),
        tasks=TasksV1(v1_http),
        flows=FlowsV1(v1_http),
    )

    if version == "v1":
        return v1

    v2 = APIBackend(
        datasets=DatasetsV2(v2_http),
        tasks=TasksV2(v2_http),
        flows=FlowsV2(v2_http),
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
