# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
from openml._api.config import settings
  
from openml._api.resources.evaluations import EvaluationsV1, EvaluationsV2
from openml.evaluations import OpenMLEvaluation
from openml.testing import TestAPIBase
from openml._api.resources.base.fallback import FallbackProxy
  
  
class TestEvaluationsV1(TestAPIBase):  
    """Tests for V1 XML API implementation of evaluations."""  
  
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
        self.resource = EvaluationsV1(self.client)
  
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


class TestEvaluationsV2(TestAPIBase): 
    """Tests for V2 JSON API implementation of evaluations."""  
  
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
        self.resource = EvaluationsV2(self.client)


class TestEvaluationsCombined(TestAPIBase):
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
        self.resource_v1 = EvaluationsV1(self.v1_client)
        self.resource_v2 = EvaluationsV2(self.v2_client)
        self.resource_fallback = FallbackProxy(self.resource_v2, self.resource_v1)