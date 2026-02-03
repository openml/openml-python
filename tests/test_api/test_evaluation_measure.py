# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api.resources.evaluation_measure import EvaluationMeasureV1API, EvaluationMeasureV2API
from openml.testing import TestAPIBase
from openml._api.resources.base.fallback import FallbackProxy
  
  
class TestEvaluationMeasureV1(TestAPIBase):  
    """Tests for V1 XML API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        self.resource = EvaluationMeasureV1API(self.http_client)
  
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
        self.client = self._get_http_client(
            server="http://localhost:8001/",
            base_url="",
            api_key="",
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        self.resource = EvaluationMeasureV2API(self.client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        measures = self.resource.list()   
        assert isinstance(measures, list) is True
        assert all(isinstance(s, str) for s in measures) is True 


class TestEvaluationMeasuresCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self.http_client
        self.v2_client = self._get_http_client(
            server="http://localhost:8001/",
            base_url="",
            api_key="",
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache,
        )
        self.resource_v1 = EvaluationMeasureV1API(self.v1_client)
        self.resource_v2 = EvaluationMeasureV2API(self.v2_client)
        self.resource_fallback = FallbackProxy(self.resource_v2, self.resource_v1)

    @pytest.mark.uses_test_server()
    def test_list_matches(self):
        output_v1 = self.resource_v1.list()
        output_v2 = self.resource_v2.list()

        assert isinstance(output_v1, list)
        assert isinstance(output_v2, list)
        assert output_v1 == output_v2

    @pytest.mark.uses_test_server()
    def test_list_fallback(self):
        output_fallback = self.resource_fallback.list()
        assert isinstance(output_fallback, list)
        assert all(isinstance(s, str) for s in output_fallback)