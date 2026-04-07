from __future__ import annotations

from .base import FlowAPI, ResourceV1API, ResourceV2API


class FlowV1API(ResourceV1API, FlowAPI):
    """Version 1 API implementation for flow resources."""


class FlowV2API(ResourceV2API, FlowAPI):
    """Version 2 API implementation for flow resources."""
