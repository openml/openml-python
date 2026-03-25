from __future__ import annotations

from .base import ResourceV1API, ResourceV2API, TaskAPI


class TaskV1API(ResourceV1API, TaskAPI):
    """Version 1 API implementation for task resources."""


class TaskV2API(ResourceV2API, TaskAPI):
    """Version 2 API implementation for task resources."""
