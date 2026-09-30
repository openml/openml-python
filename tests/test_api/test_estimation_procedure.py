# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api import EstimationProcedureV1API, EstimationProcedureV2API
from openml.exceptions import OpenMLNotSupportedError
from openml.estimation_procedures import OpenMLEstimationProcedure


@pytest.fixture
def estimation_procedure_v1(http_client_v1, minio_client) -> EstimationProcedureV1API:
    return EstimationProcedureV1API(http=http_client_v1, minio=minio_client)


@pytest.fixture
def estimation_procedure_v2(http_client_v2, minio_client) -> EstimationProcedureV2API:
    return EstimationProcedureV2API(http=http_client_v2, minio=minio_client)


@pytest.mark.test_server()
def test_v1_list(estimation_procedure_v1):
    details = estimation_procedure_v1.list()
    
    assert isinstance(details, list)
    assert len(details) > 0
    assert all(isinstance(d, OpenMLEstimationProcedure) for d in details)


@pytest.mark.test_server()
def test_v2_list(estimation_procedure_v2):
    with pytest.raises(OpenMLNotSupportedError):
        estimation_procedure_v2.list()
