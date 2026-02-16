# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest    
from openml._api.resources.estimation_procedure import EstimationProcedureV1API, EstimationProcedureV2API
from openml.testing import TestAPIBase
from openml.enums import APIVersion  
from openml.exceptions import OpenMLNotSupportedError

class TestEstimationProcedureAPIBase(TestAPIBase):
    resource: EstimationProcedureV1API | EstimationProcedureV2API

    def _list(self):
        procedures = self.resource.list()
        
        assert isinstance(procedures, list)
        assert len(procedures) > 0
        assert all(isinstance(p, str) for p in procedures)
    
    def _get_details(self):
        details = self.resource._get_details()
        
        assert isinstance(details, list)
        assert len(details) > 0
        assert all(isinstance(d, dict) for d in details)

        assert all("id" in d for d in details)
        assert all("name" in d for d in details)
        assert all("task_type_id" in d for d in details)

class TestEstimationProcedureV1API(TestEstimationProcedureAPIBase):  
    """Tests for V1 XML API implementation of estimation procedures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V1]
        self.resource = EstimationProcedureV1API(http_client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        self._list()

    
    @pytest.mark.uses_test_server()
    def test_get_details(self):
        self._get_details()


class TestEstimationProcedureV2API(TestEstimationProcedureAPIBase):
    """Tests for V2 JSON API implementation of estimation procedures."""   
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V2]
        self.resource = EstimationProcedureV2API(http_client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        self._list()
    
    @pytest.mark.uses_test_server()
    def test_get_details(self):
        with pytest.raises(OpenMLNotSupportedError):
            self._get_details()
        

class TestEstimationProcedureCombinedAPI(TestEstimationProcedureAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self.http_clients[APIVersion.V1]
        self.v2_client = self.http_clients[APIVersion.V2]
        self.resource_v1 = EstimationProcedureV1API(self.v1_client)
        self.resource_v2 = EstimationProcedureV2API(self.v2_client)

    @pytest.mark.uses_test_server()
    def test_list_matches(self):
        output_v1 = self.resource_v1.list()
        output_v2 = self.resource_v2.list()

        assert isinstance(output_v1, list)
        assert isinstance(output_v2, list)
        assert output_v1 == output_v2