# License: BSD 3-Clause
from __future__ import annotations

import pytest

from openml._api import EstimationProcedureV1API, EstimationProcedureV2API
from openml.estimation_procedures import OpenMLEstimationProcedure
from openml.exceptions import OpenMLNotSupportedError


@pytest.fixture
def estimation_procedure_v1(http_client_v1, s3_client) -> EstimationProcedureV1API:
    return EstimationProcedureV1API(http=http_client_v1, s3=s3_client)


@pytest.fixture
def estimation_procedure_v2(http_client_v2, s3_client) -> EstimationProcedureV2API:
    return EstimationProcedureV2API(http=http_client_v2, s3=s3_client)


@pytest.mark.test_server
def test_v1_list(estimation_procedure_v1):
    details = estimation_procedure_v1.list()

    assert isinstance(details, list)
    assert len(details) > 0
    assert all(isinstance(d, OpenMLEstimationProcedure) for d in details)


@pytest.mark.test_server
def test_v2_list(estimation_procedure_v2):
    with pytest.raises(OpenMLNotSupportedError):
        estimation_procedure_v2.list()
