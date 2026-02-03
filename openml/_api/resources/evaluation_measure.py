from __future__ import annotations

from .base import EvaluationMeasureAPI, ResourceV1API, ResourceV2API


class EvaluationMeasureV1API(ResourceV1API, EvaluationMeasureAPI):
    pass


class EvaluationMeasureV2API(ResourceV2API, EvaluationMeasureAPI):
    pass
