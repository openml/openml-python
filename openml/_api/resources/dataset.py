from __future__ import annotations

from .base import DatasetAPI, ResourceV1API, ResourceV2API


class DatasetV1API(ResourceV1API, DatasetAPI):
    pass


class DatasetV2API(ResourceV2API, DatasetAPI):
    pass
