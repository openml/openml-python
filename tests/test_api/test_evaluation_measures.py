# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
  
from openml._api.runtime.core import build_backend   
from openml.testing import TestBase  
  
  
class TestEvaluationMeasuresV1(TestBase):  
    """Tests for V1 XML API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v1', strict=True)
        self.api = backend.evaluation_measures 
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        measures = self.api.list()   
        assert isinstance(measures, list) is True
        assert all(isinstance(s, str) for s in measures) is True  


class TestEvaluationMeasuresV2(TestBase): 
    """Tests for V2 JSON API implementation of evaluation measures."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v2', strict=True)
        self.api = backend.evaluation_measures 
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        measures = self.api.list()   
        assert isinstance(measures, list) is True
        assert all(isinstance(s, str) for s in measures) is True 
  