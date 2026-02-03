# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api.resources.evaluation import EvaluationV1API, EvaluationV2API
from openml.evaluations import OpenMLEvaluation
from openml.testing import TestAPIBase
from openml._api.resources.base.fallback import FallbackProxy
  
  
class TestEvaluationV1(TestAPIBase):  
    """Tests for V1 XML API implementation of evaluations."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        self.resource = EvaluationV1API(self.http_client)
  
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
        self.client = self._get_http_client(
            server="",
            base_url="",
            api_key="",
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        self.resource = EvaluationV2API(self.client)


class TestEvaluationsCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self.http_client
        self.v2_client = self._get_http_client(
            server="",
            base_url="",
            api_key="",
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        self.resource_v1 = EvaluationV1API(self.v1_client)
        self.resource_v2 = EvaluationV2API(self.v2_client)
        self.resource_fallback = FallbackProxy(self.resource_v2, self.resource_v1)

    @pytest.mark.uses_test_server()
    def test_list_fallback(self):
        output_fallback = self.resource_fallback.list(
            function="predictive_accuracy",
            limit=10,
            offset=0,
        )
        assert isinstance(output_fallback, list)
        assert len(output_fallback) == 10
        assert all(isinstance(e, OpenMLEvaluation) for e in output_fallback)