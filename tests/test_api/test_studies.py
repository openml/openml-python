# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
  
from openml._api.runtime.core import build_backend
from openml.study import OpenMLStudy
from openml.testing import TestBase  
  
  
class TestStudiesV1(TestBase):  
    """Tests for V1 XML API implementation of studies."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v1', strict=True)
        self.api = backend.studies
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        studies = self.api.list(
            limit=10,
            offset=0,
        )
        
        assert isinstance(studies, list)
        assert len(studies) == 10
        assert all(isinstance(e, OpenMLStudy) for e in studies)


class TestStudiesV2(TestBase): 
    """Tests for V2 JSON API implementation of studies."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        backend = build_backend('v2', strict=True)
        self.api = backend.studies