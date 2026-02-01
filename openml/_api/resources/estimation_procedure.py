from __future__ import annotations

from openml._api.resources.base import EstimationProcedureAPI, ResourceV1API, ResourceV2API


class EstimationProcedureV1API(ResourceV1API, EstimationProcedureAPI):
    pass


class EstimationProcedureV2API(ResourceV2API, EstimationProcedureAPI):
    pass
