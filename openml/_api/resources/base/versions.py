from __future__ import annotations

from openml._api.resources.base import APIVersion, ResourceAPI


class ResourceV1(ResourceAPI):
    api_version: APIVersion | None = APIVersion.V1

    def delete(self) -> None:
        pass

    def publish(self) -> None:
        pass


class ResourceV2(ResourceAPI):
    api_version: APIVersion | None = APIVersion.V2

    def delete(self) -> None:
        self._raise_not_implemented_error("delete")

    def publish(self) -> None:
        self._raise_not_implemented_error("publish")
