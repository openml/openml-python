from __future__ import annotations
from pathlib import Path
from time import time

from openml import OpenMLDataset
import pytest
import pandas as pd
from openml._api.resources.base.fallback import FallbackProxy
from openml.testing import TestAPIBase
from openml._api.resources.dataset import DatasetV1API, DatasetV2API


class TestDatasetV1API(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.client = self._get_http_client(
            server=self.server,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache
        )
        self.dataset = DatasetV1API(self.client,self.minio_client)
    
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

    @pytest.mark.uses_test_server()
    def test_status_update(self):
        dataset = OpenMLDataset(
            f"TEST-{str(time())}-UploadTestWithURL",
            "test",
            "ARFF",
            version=1,
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff",
        )
        file_elements= dict()
        file_elements["description"] = dataset._to_xml()
        dataset_id = self.dataset.publish(path="data", files=file_elements)
        self.dataset_id = dataset_id
        # admin key for test server (only admins can activate datasets.
        # all users can deactivate their own datasets)
        self.api_key = self.dataset._http.api_key = self.admin_key

        status = "deactivated"
        self.dataset.status_update(dataset_id,status)
        result = self.dataset.list(limit=1,offset=0,data_id=[dataset_id], status="all")
        result = result.to_dict(orient="index")
        assert result[dataset_id]["status"] == status
        
        status = "active"
        self.dataset.status_update(dataset_id,status)
        result = self.dataset.list(limit=1,offset=0,data_id=[dataset_id], status="all")
        result = result.to_dict(orient="index")
        assert result[dataset_id]["status"] == status

        assert self.dataset.delete(dataset_id)

    @pytest.mark.uses_test_server()
    def test_edit(self):pass

    @pytest.mark.uses_test_server()
    def test_fork(self):pass

    @pytest.mark.uses_test_server()
    def test_list_qualities(self):
        output = self.dataset.list_qualities()
        assert len(output) == 19
        assert isinstance(output[0],str)

    @pytest.mark.uses_test_server()
    def test_feature_add_remove_ontology(self):
        did = 11
        fid = 0
        ontology = "https://www.openml.org/unittest/" + str(time())
        output = self.dataset.feature_add_ontology(did,fid,ontology)
        assert output

        output = self.dataset.feature_remove_ontology(did,fid,ontology)
        assert output

    @pytest.mark.uses_test_server()
    def test_add_remove_topic(self):pass


class TestDatasetV2API(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.client = self._get_http_client(
            server="http://127.0.0.1:8001/",
            base_url="",
            api_key="",
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache
        )
        self.dataset = DatasetV2API(self.client,self.minio_client)
    
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

    @pytest.mark.uses_test_server()
    def test_list_qualities(self):
        output = self.dataset.list_qualities()
        assert len(output) == 107
        assert isinstance(output[0],str)


class TestDatasetsCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
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
            api_key="",
            timeout=self.timeout,
            retries=self.retries,
            retry_policy=self.retry_policy,
            cache=self.cache
        )
        self.dataset_v1 = DatasetV1API(self.v1_client,self.minio_client)
        self.dataset_v2 = DatasetV2API(self.v2_client,self.minio_client)
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
        output_v1 = self.dataset_v1.get_features(3)
        output_v2 = self.dataset_v2.get_features(3)

        assert output_v1.keys() == output_v2.keys()
        # would not be same if v1 has ontology
        assert output_v1 == output_v2

    @pytest.mark.uses_test_server()
    def test_get_features_fallback(self):
        output_fallback = self.dataset_fallback.get_features(2)

        assert isinstance(output_fallback,dict)
        assert len(output_fallback.keys()) == 37

    @pytest.mark.uses_test_server()
    def test_get_qualities_matches(self):
        #TODO Qualities in local python server and test server differ
        output_v1 = self.dataset_v1.get_qualities(2)
        output_v2 = self.dataset_v2.get_qualities(2)
        assert output_v1['AutoCorrelation'] == output_v2['AutoCorrelation']


    @pytest.mark.uses_test_server()
    def test_get_qualities_fallback(self):
        #TODO Qualities in local python server and test server differ
        output_fallback = self.dataset_fallback.get_qualities(2)

        assert isinstance(output_fallback,dict)

    @pytest.mark.uses_test_server()
    def test_list_qualities_matches(self):
        #TODO Qualities in local python server and test server differ
        output_v1 = self.dataset_v1.list_qualities()
        output_v2 = self.dataset_v2.list_qualities()

        assert "AutoCorrelation" in output_v1
        assert "AutoCorrelation" in output_v2


    @pytest.mark.uses_test_server()
    def test_list_qualities_fallback(self):
        #TODO Qualities in local python server and test server differ
        output_fallback = self.dataset_fallback.list_qualities()

        assert isinstance(output_fallback,list)

    @pytest.mark.uses_test_server()
    def test_status_update_fallback(self):
        dataset = OpenMLDataset(
            f"TEST-{str(time())}-UploadTestWithURL",
            "test",
            "ARFF",
            version=1,
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff",
        )
        file_elements= dict()
        file_elements["description"] = dataset._to_xml()
        dataset_id = self.dataset_fallback.publish(path="data", files=file_elements)
        self.dataset_id = dataset_id
        self.api_key = self.dataset_fallback._http.api_key = self.admin_key

        self.dataset_fallback.status_update(dataset_id,"deactivated")
        self.dataset_fallback.status_update(dataset_id,"active")

        assert self.dataset_fallback.delete(dataset_id)

    @pytest.mark.uses_test_server()
    def test_edit_fallback(self):pass

    @pytest.mark.uses_test_server()
    def test_fork_fallback(self):pass

    @pytest.mark.uses_test_server()
    def test_feature_add_remove_ontology_fallback(self):
        ontology = "https://www.openml.org/unittest/" + str(time())
        output_fallback_add = self.dataset_fallback.feature_add_ontology(
            11, 0,ontology)
        assert output_fallback_add
        output_fallback_remove = self.dataset_fallback.feature_remove_ontology(
            11, 0,ontology)
        assert output_fallback_remove
    
    @pytest.mark.uses_test_server()
    def test_add_remove_topic_fallback(self):pass

