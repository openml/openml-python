from __future__ import annotations

from .base import ResourceV1API, ResourceV2API, SetupAPI


class SetupV1API(ResourceV1API, SetupAPI):
    pass


class SetupV2API(ResourceV2API, SetupAPI):
    pass
