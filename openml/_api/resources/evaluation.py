from __future__ import annotations

from .base import EvaluationAPI, ResourceV1API, ResourceV2API


class EvaluationV1API(ResourceV1API, EvaluationAPI):
    """Version 1 API implementation for evaluation resources."""


class EvaluationV2API(ResourceV2API, EvaluationAPI):
    """Version 2 API implementation for evaluation resources."""
