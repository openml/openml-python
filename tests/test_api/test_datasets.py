from __future__ import annotations
from pathlib import Path

import pytest
import pandas as pd
from openml._api.clients.minio import MinIOClient
from openml._api.resources.base.fallback import FallbackProxy
from openml.testing import TestAPIBase
from openml._api.resources.datasets import DatasetsV1, DatasetsV2


class TestDatasetsV1(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.minio_client = MinIOClient()
        self.client = self._get_http_client(
            server=self.server,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache
        )
        self.dataset = DatasetsV1(self.client,self.minio_client)
    
    @pytest.mark.uses_test_server()
    def test_get(self):
        output = self.dataset.get(2)
        assert output.dataset_id == 2

    @pytest.mark.uses_test_server()
    def test_list(self):
        output =self.dataset.list(limit=2, offset=0, status="active")
        assert not output.empty
        assert output.shape[0] == 2
        assert (output["status"].nunique() == 1)
        assert (output["status"].unique()[0] == "active")

    @pytest.mark.uses_test_server()
    def test_download_arff_from_get(self):
        output = self.dataset.get(2,download_data=True)

        assert output.data_file != None
        assert Path(output.data_file).exists()

    @pytest.mark.uses_test_server()
    def test_download_qualities_from_get(self):
        output = self.dataset.get(2,download_qualities=True)

        assert output._qualities is not None
    
    @pytest.mark.uses_test_server()
    def test_download_features_from_get(self):
        output = self.dataset.get(2,download_features_meta_data=True)

        assert output._features is not None

    @pytest.mark.uses_test_server()
    def test_get_features(self):
        output = self.dataset.get_features(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 37

    @pytest.mark.uses_test_server()
    def test_get_qualities(self):
        output = self.dataset.get_qualities(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 19



class TestDatasetsV2(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.minio_client = MinIOClient()
        self.client = self._get_http_client(
            server="http://127.0.0.1:8001/",
            base_url="",
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache
        )
        self.dataset = DatasetsV2(self.client,self.minio_client)
    
    @pytest.mark.uses_test_server()
    def test_get(self):
        output= self.dataset.get(2)
        assert output.dataset_id == 2
    
    @pytest.mark.uses_test_server()
    def test_list(self):
        output =self.dataset.list(limit=2, offset=0, status="active")
        assert not output.empty
        assert output.shape[0] == 2
        assert (output["status"].nunique() == 1)
        assert (output["status"].unique()[0] == "active")
    
    @pytest.mark.uses_test_server()
    def test_download_arff_from_get(self):
        output = self.dataset.get(2,download_data=True)
        
        assert output.data_file != None
        assert Path(output.data_file).exists()

    @pytest.mark.uses_test_server()
    def test_download_qualities_from_get(self):
        output = self.dataset.get(2,download_qualities=True)

        assert output._qualities is not None
    
    @pytest.mark.uses_test_server()
    def test_download_features_from_get(self):
        output = self.dataset.get(2,download_features_meta_data=True)

        assert output._features is not None

    @pytest.mark.uses_test_server()
    def test_get_features(self):
        output = self.dataset.get_features(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 37

    @pytest.mark.uses_test_server()
    def test_get_qualities(self):
        output = self.dataset.get_qualities(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 107


class TestDatasetsCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.minio_client = MinIOClient()
        self.v1_client = self._get_http_client(
			server=self.server,
			base_url=self.base_url,
			api_key=self.api_key,
			timeout=self.timeout,
			retries=self.retries,
			retry_policy=self.retry_policy,
            cache=self.cache
		)
        self.v2_client = self._get_http_client(
            server="http://127.0.0.1:8001/",
            base_url="",
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache
        )
        self.dataset_v1 = DatasetsV1(self.v1_client,self.minio_client)
        self.dataset_v2 = DatasetsV2(self.v2_client,self.minio_client)
        self.dataset_fallback = FallbackProxy(self.dataset_v1,self.dataset_v2)
    
    @pytest.mark.uses_test_server()
    def test_get_matches(self):
        output_v1 = self.dataset_v1.get(2)
        output_v2 = self.dataset_v2.get(2)

        assert output_v1.dataset_id == output_v2.dataset_id
        assert output_v1.name == output_v2.name
        assert output_v1.data_file is None
        assert output_v1.data_file == output_v2.data_file
    
    @pytest.mark.uses_test_server()
    def test_get_fallback(self):
        output_fallback =  self.dataset_fallback.get(2)
        assert output_fallback.dataset_id == 2

    #TODO list has different structure compared to v1
    @pytest.mark.uses_test_server()
    def test_list_matches(self):
        output_v1 = self.dataset_v1.list(limit=2, offset=1)
        output_v2 = self.dataset_v2.list(limit=2, offset=1)

        pd.testing.assert_series_equal(output_v1["did"],output_v2["did"])
    
    @pytest.mark.uses_test_server()
    def test_list_fallback(self):
        output_fallback =self.dataset_fallback.list(limit=2, offset=0,data_id=[2,3])

        assert not output_fallback.empty
        assert output_fallback.shape[0] == 2
        assert set(output_fallback["did"]) == {2, 3}

    @pytest.mark.uses_test_server()
    def test_get_features_matches(self):
        output_v1 = self.dataset_v1.get_features(2)
        output_v2 = self.dataset_v2.get_features(2)

        assert output_v1.keys() == output_v2.keys()
        assert output_v1 == output_v2

    @pytest.mark.uses_test_server()
    def test_get_features_fallback(self):
        output_fallback = self.dataset_fallback.get_features(2)

        assert isinstance(output_fallback,dict)
        assert len(output_fallback.keys()) == 37

    @pytest.mark.uses_test_server()
    def test_get_qualities_matches(self):
        output_v1 = self.dataset_v1.get_qualities(2)
        output_v2 = self.dataset_v2.get_qualities(2)

        #TODO Qualities in local python server and test server differ

    @pytest.mark.uses_test_server()
    def test_get_qualities_fallback(self):
        output_fallback = self.dataset_fallback.get_qualities(2)

        assert isinstance(output_fallback,dict)
        #TODO Qualities in local python server and test server differ