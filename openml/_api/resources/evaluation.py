from __future__ import annotations

from openml._api.resources.base import EvaluationAPI, ResourceV1API, ResourceV2API


class EvaluationV1API(ResourceV1API, EvaluationAPI):
    pass


class EvaluationV2API(ResourceV2API, EvaluationAPI):
    pass
