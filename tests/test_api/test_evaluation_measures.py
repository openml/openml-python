# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
from openml._api.config import settings
  
from openml._api.resources.evaluation_measures import EvaluationMeasuresV1, EvaluationMeasuresV2
from openml.testing import TestAPIBase
from openml._api.resources.base.fallback import FallbackProxy
  
  
class TestEvaluationMeasuresV1(TestAPIBase):  
    """Tests for V1 XML API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        self.client = self._get_http_client(
            server=settings.api.v1.server,
            base_url=settings.api.v1.base_url,
            api_key=settings.api.v1.api_key,
            timeout=settings.api.v1.timeout,
            retries=settings.connection.retries,
            retry_policy=settings.connection.retry_policy,
        )
        self.resource = EvaluationMeasuresV1(self.client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        measures = self.resource.list()   
        assert isinstance(measures, list) is True
        assert all(isinstance(s, str) for s in measures) is True  


class TestEvaluationMeasuresV2(TestAPIBase): 
    """Tests for V2 JSON API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        self.client = self._get_http_client(
            server=settings.api.v2.server,
            base_url=settings.api.v2.base_url,
            api_key=settings.api.v2.api_key,
            timeout=settings.api.v2.timeout,
            retries=settings.connection.retries,
            retry_policy=settings.connection.retry_policy,
        )
        self.resource = EvaluationMeasuresV2(self.client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        measures = self.resource.list()   
        assert isinstance(measures, list) is True
        assert all(isinstance(s, str) for s in measures) is True 


class TestEvaluationMeasuresCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self._get_http_client(
            server=settings.api.v1.server,
            base_url=settings.api.v1.base_url,
            api_key=settings.api.v1.api_key,
            timeout=settings.api.v1.timeout,
            retries=settings.connection.retries,
            retry_policy=settings.connection.retry_policy,
        )
        self.v2_client = self._get_http_client(
            server=settings.api.v2.server,
            base_url=settings.api.v2.base_url,
            api_key=settings.api.v2.api_key,
            timeout=settings.api.v2.timeout,
            retries=settings.connection.retries,
            retry_policy=settings.connection.retry_policy,
        )
        self.resource_v1 = EvaluationMeasuresV1(self.v1_client)
        self.resource_v2 = EvaluationMeasuresV2(self.v2_client)
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