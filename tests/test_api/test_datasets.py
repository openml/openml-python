from __future__ import annotations

import pytest
import pandas as pd
import requests
from openml.testing import TestBase
from openml._api import api_context
from openml._api.resources.datasets import DatasetsV1, DatasetsV2

class TestDatasetsEndpoints(TestBase):
    def setUp(self):
        super().setUp()
        self.v1_api = DatasetsV1(
            api_context.backend.datasets._http,
            api_context.backend.datasets._minio
        )
        self.v2_api = DatasetsV2(
            api_context.backend.datasets._http,
            api_context.backend.datasets._minio
        )
    
