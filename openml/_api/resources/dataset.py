from __future__ import annotations

from .base import DatasetAPI, ResourceV1API, ResourceV2API


class DatasetV1API(ResourceV1API, DatasetAPI):
    """Version 1 API implementation for dataset resources."""


class DatasetV2API(ResourceV2API, DatasetAPI):
    """Version 2 API implementation for dataset resources."""
