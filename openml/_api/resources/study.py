from __future__ import annotations

from .base import ResourceV1API, ResourceV2API, StudyAPI


class StudyV1API(ResourceV1API, StudyAPI):
    """Version 1 API implementation for study resources."""


class StudyV2API(ResourceV2API, StudyAPI):
    """Version 2 API implementation for study resources."""
