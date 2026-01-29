# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
from openml._api.config import settings
  
from openml._api.resources.setups import SetupsV1, SetupsV2
from openml.setups.setup import OpenMLSetup
from openml.testing import TestAPIBase
from openml._api.resources.base.fallback import FallbackProxy

  
  
class TestSetupsV1(TestAPIBase):  
    """Tests for V1 XML API implementation of setups."""  
  
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
        self.resource = SetupsV1(self.client)
  
    @pytest.mark.uses_test_server()
    def test_list(self):
        setups = self.resource.list(limit=10, offset=0)
        
        assert isinstance(setups, list)
        assert len(setups) > 0
        assert all(isinstance(s, OpenMLSetup) for s in setups)

    
    def test_get(self):
        setup_id = 1
        xml_content, setup = self.resource.get(setup_id)
        
        assert isinstance(xml_content, str)
        assert isinstance(setup, OpenMLSetup)
        assert setup.setup_id == setup_id


class TestSetupsV2(TestAPIBase): 
    """Tests for V2 JSON API implementation of setups."""  
  
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
        self.resource = SetupsV2(self.client)

class TestSetupsCombined(TestAPIBase):
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
		self.resource_v1 = SetupsV1(self.client)
		self.resource_v2 = SetupsV2(self.client)
		self.resource_fallback = FallbackProxy(self.resource_v2, self.resource_v1)
