# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
  
from openml._api.runtime.core import build_backend
from openml.evaluations import OpenMLEvaluation
from openml.testing import TestBase  
  
  
class TestEvaluationsV1(TestBase):  
    """Tests for V1 XML API implementation of evaluations."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v1', strict=True)
        self.api = backend.evaluations
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        evaluations = self.api.list(
            function="predictive_accuracy",
            limit=10,
            offset=0,
        )
        
        assert isinstance(evaluations, list)
        assert len(evaluations) == 10
        assert all(isinstance(e, OpenMLEvaluation) for e in evaluations)


class TestEvaluationsV2(TestBase): 
    """Tests for V2 JSON API implementation of evaluations."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v2', strict=True)
        self.api = backend.evaluations
  