# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api import EvaluationMeasureV1API, EvaluationMeasureV2API

@pytest.fixture
def evaluation_measure_v1(http_client_v1, minio_client) -> EvaluationMeasureV1API:
    return EvaluationMeasureV1API(http=http_client_v1, minio=minio_client)


@pytest.fixture
def evaluation_measure_v2(http_client_v2, minio_client) -> EvaluationMeasureV2API:
    return EvaluationMeasureV2API(http=http_client_v2, minio=minio_client)


@pytest.mark.uses_test_server()    
def test_v1_list(evaluation_measure_v1):
    measures = evaluation_measure_v1.list()   
    assert isinstance(measures, list) is True
    assert all(isinstance(s, str) for s in measures) is True

@pytest.mark.uses_test_server()    
def test_v2_list(evaluation_measure_v2):
    measures = evaluation_measure_v2.list()   
    assert isinstance(measures, list) is True
    assert all(isinstance(s, str) for s in measures) is True

@pytest.mark.uses_test_server()
def test_list_matches(evaluation_measure_v1,evaluation_measure_v2):
    output_v1 = evaluation_measure_v1.list()
    output_v2 = evaluation_measure_v2.list()

    assert isinstance(output_v1, list)
    assert isinstance(output_v2, list)
    assert output_v1 == output_v2