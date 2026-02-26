# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api import EvaluationV1API, EvaluationV2API
from openml.evaluations import OpenMLEvaluation
from openml.exceptions import OpenMLNotSupportedError  


@pytest.fixture
def evaluation_v1(http_client_v1, minio_client) -> EvaluationV1API:
    return EvaluationV1API(http=http_client_v1, minio=minio_client)

@pytest.fixture
def evaluation_v2(http_client_v2, minio_client) -> EvaluationV2API:
    return EvaluationV2API(http=http_client_v2, minio=minio_client)


@pytest.mark.uses_test_server()
def test_v1_list(evaluation_v1):
    evaluations = evaluation_v1.list(
        function="predictive_accuracy",
        limit=10,
        offset=0,
    )
    
    assert isinstance(evaluations, list)
    assert len(evaluations) == 10
    assert all(isinstance(e, OpenMLEvaluation) for e in evaluations)
  
    
@pytest.mark.uses_test_server()
def test_v2_list(evaluation_v2):
    with pytest.raises(OpenMLNotSupportedError):
        evaluation_v2.list(
        function="predictive_accuracy",
        limit=10,
        offset=0,
    )
            