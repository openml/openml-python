from __future__ import annotations

from openml._api.resources.base import DatasetAPI, ResourceV1API, ResourceV2API


class DatasetV1API(ResourceV1API, DatasetAPI):
    pass


class DatasetV2API(ResourceV2API, DatasetAPI):
    pass
