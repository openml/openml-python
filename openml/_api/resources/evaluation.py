from __future__ import annotations

from .base import EvaluationAPI, ResourceV1API, ResourceV2API


class EvaluationV1API(ResourceV1API, EvaluationAPI):
    pass


class EvaluationV2API(ResourceV2API, EvaluationAPI):
    pass
