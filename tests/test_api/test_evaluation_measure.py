# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api.resources.evaluation_measure import EvaluationMeasureV1API, EvaluationMeasureV2API
from openml.testing import TestAPIBase
from openml.enums import APIVersion  
  
class TestEvaluationMeasureV1(TestAPIBase):  
    """Tests for V1 XML API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V1]
        self.resource = EvaluationMeasureV1API(http_client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        measures = self.resource.list()   
        assert isinstance(measures, list) is True
        assert all(isinstance(s, str) for s in measures) is True  


class TestEvaluationMeasureV2(TestAPIBase): 
    """Tests for V2 JSON API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V2]
        self.resource = EvaluationMeasureV2API(http_client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        measures = self.resource.list()   
        assert isinstance(measures, list) is True
        assert all(isinstance(s, str) for s in measures) is True 


class TestEvaluationMeasuresCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self.http_clients[APIVersion.V1]
        self.v2_client = self.http_clients[APIVersion.V2]
        self.resource_v1 = EvaluationMeasureV1API(self.v1_client)
        self.resource_v2 = EvaluationMeasureV2API(self.v2_client)

    @pytest.mark.uses_test_server()
    def test_list_matches(self):
        output_v1 = self.resource_v1.list()
        output_v2 = self.resource_v2.list()

        assert isinstance(output_v1, list)
        assert isinstance(output_v2, list)
        assert output_v1 == output_v2