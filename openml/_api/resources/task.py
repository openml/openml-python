from __future__ import annotations

from openml._api.resources.base import ResourceV1API, ResourceV2API, TaskAPI


class TaskV1API(ResourceV1API, TaskAPI):
    pass


class TaskV2API(ResourceV2API, TaskAPI):
    pass
