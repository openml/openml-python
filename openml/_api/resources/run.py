from __future__ import annotations

from .base import ResourceV1API, ResourceV2API, RunAPI


class RunV1API(ResourceV1API, RunAPI):
    """Version 1 API implementation for run resources."""


class RunV2API(ResourceV2API, RunAPI):
    """Version 2 API implementation for run resources."""
