# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
  
from openml._api.runtime.core import build_backend
from openml.testing import TestBase  
  
  
class TestEstimationProceduresV1(TestBase):  
    """Tests for V1 XML API implementation of estimation procedures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v1', strict=True)
        self.api = backend.estimation_procedures
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        procedures = self.api.list()
        
        assert isinstance(procedures, list)
        assert len(procedures) > 0
        assert all(isinstance(p, str) for p in procedures)

    
    @pytest.mark.uses_test_server()
    def test_get_details(self):
        details = self.api._get_details()
        
        assert isinstance(details, list)
        assert len(details) > 0
        assert all(isinstance(d, dict) for d in details)

        assert all("id" in d for d in details)
        assert all("name" in d for d in details)
        assert all("task_type_id" in d for d in details)


class TestEstimationProceduresV2(TestBase):
    """Tests for V2 JSON API implementation of estimation procedures."""   
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v2', strict=True)
        self.api = backend.estimation_procedures
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        procedures = self.api.list()
        
        assert isinstance(procedures, list)
        assert len(procedures) > 0
        assert all(isinstance(p, str) for p in procedures)
