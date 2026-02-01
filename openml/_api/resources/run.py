from __future__ import annotations

from openml._api.resources.base import ResourceV1API, ResourceV2API, RunAPI


class RunV1API(ResourceV1API, RunAPI):
    pass


class RunV2API(ResourceV2API, RunAPI):
    pass
