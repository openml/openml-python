# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api.resources.evaluation import EvaluationV1API, EvaluationV2API
from openml.evaluations import OpenMLEvaluation
from openml.testing import TestAPIBase
from openml.enums import APIVersion  
from openml.exceptions import OpenMLNotSupportedError  
  
class TestEvaluationV1(TestAPIBase):  
    """Tests for V1 XML API implementation of evaluations."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V1]
        self.resource = EvaluationV1API(http_client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        evaluations = self.resource.list(
            function="predictive_accuracy",
            limit=10,
            offset=0,
        )
        
        assert isinstance(evaluations, list)
        assert len(evaluations) == 10
        assert all(isinstance(e, OpenMLEvaluation) for e in evaluations)


class TestEvaluationV2(TestAPIBase): 
    """Tests for V2 JSON API implementation of evaluations."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V2]
        self.resource = EvaluationV2API(http_client)

    @pytest.mark.uses_test_server()
    def test_list(self):
        with pytest.raises(OpenMLNotSupportedError):
            self.resource.list(
                function="predictive_accuracy",
                limit=10,
                offset=0,
            )
            


class TestEvaluationsCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self.http_clients[APIVersion.V1]
        self.v2_client = self.http_clients[APIVersion.V2]
        self.resource_v1 = EvaluationV1API(self.v1_client)
        self.resource_v2 = EvaluationV2API(self.v2_client)
