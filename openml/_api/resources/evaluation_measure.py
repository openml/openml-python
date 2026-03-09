from __future__ import annotations

from .base import EvaluationMeasureAPI, ResourceV1API, ResourceV2API


class EvaluationMeasureV1API(ResourceV1API, EvaluationMeasureAPI):
    """Version 1 API implementation for evaluation measure resources."""


class EvaluationMeasureV2API(ResourceV2API, EvaluationMeasureAPI):
    """Version 2 API implementation for evaluation measure resources."""
