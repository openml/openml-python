# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api.resources.estimation_procedure import EstimationProcedureV1API, EstimationProcedureV2API
from openml.testing import TestAPIBase
from openml._api.resources.base.fallback import FallbackProxy
  
  
class TestEstimationProceduresV1(TestAPIBase):  
    """Tests for V1 XML API implementation of estimation procedures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        self.resource = EstimationProcedureV1API(self.http_client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        procedures = self.resource.list()
        
        assert isinstance(procedures, list)
        assert len(procedures) > 0
        assert all(isinstance(p, str) for p in procedures)

    
    @pytest.mark.uses_test_server()
    def test_get_details(self):
        details = self.resource._get_details()
        
        assert isinstance(details, list)
        assert len(details) > 0
        assert all(isinstance(d, dict) for d in details)

        assert all("id" in d for d in details)
        assert all("name" in d for d in details)
        assert all("task_type_id" in d for d in details)


class TestEstimationProceduresV2(TestAPIBase):
    """Tests for V2 JSON API implementation of estimation procedures."""   
  
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
        self.resource = EstimationProcedureV2API(self.client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        procedures = self.resource.list()
        
        assert isinstance(procedures, list)
        assert len(procedures) > 0
        assert all(isinstance(p, str) for p in procedures)


class TestEstimationProceduresCombined(TestAPIBase):
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
        self.resource_v1 = EstimationProcedureV1API(self.v1_client)
        self.resource_v2 = EstimationProcedureV2API(self.v2_client)
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
        assert len(output_fallback) > 0
        assert all(isinstance(p, str) for p in output_fallback)