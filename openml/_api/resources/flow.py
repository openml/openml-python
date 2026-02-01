from __future__ import annotations

from openml._api.resources.base import FlowAPI, ResourceV1API, ResourceV2API


class FlowV1API(ResourceV1API, FlowAPI):
    pass


class FlowV2API(ResourceV2API, FlowAPI):
    pass
