from __future__ import annotations

from .base import ResourceV1API, ResourceV2API, StudyAPI


class StudyV1API(ResourceV1API, StudyAPI):
    pass


class StudyV2API(ResourceV2API, StudyAPI):
    pass
