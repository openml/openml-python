from __future__ import annotations

from .base import EstimationProcedureAPI, ResourceV1API, ResourceV2API


class EstimationProcedureV1API(ResourceV1API, EstimationProcedureAPI):
    """Version 1 API implementation for estimation procedure resources."""


class EstimationProcedureV2API(ResourceV2API, EstimationProcedureAPI):
    """Version 2 API implementation for estimation procedure resources."""
