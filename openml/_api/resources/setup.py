from __future__ import annotations

from .base import ResourceV1API, ResourceV2API, SetupAPI


class SetupV1API(ResourceV1API, SetupAPI):
    """Version 1 API implementation for setup resources."""


class SetupV2API(ResourceV2API, SetupAPI):
    """Version 2 API implementation for setup resources."""
