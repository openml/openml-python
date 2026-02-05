from __future__ import annotations
from pathlib import Path
import time

from openml import OpenMLDataset
import pytest
import pandas as pd
from openml._api.resources.base.fallback import FallbackProxy
from openml.enums import APIVersion
from openml.testing import TestAPIBase
from openml._api.resources.dataset import DatasetV1API, DatasetV2API


@pytest.mark.uses_test_server()
class TestDatasetV1API(TestAPIBase):
    def setUp(self):
        super().setUp()
        http_client_v1 = self.http_clients[APIVersion.V1]
        self.dataset = DatasetV1API(http_client_v1,self.minio_client)


    def test_get(self):
        output = self.dataset.get(2)
        assert output.dataset_id == 2


    def test_list(self):
        output =self.dataset.list(limit=2, offset=0, status="active")
        assert not output.empty
        assert output.shape[0] == 2
        assert (output["status"].nunique() == 1)
        assert (output["status"].unique()[0] == "active")


    def test_download_arff_from_get(self):
        output = self.dataset.get(2,download_data=True)

        assert output.data_file != None
        assert Path(output.data_file).exists()


    def test_download_qualities_from_get(self):
        output = self.dataset.get(2,download_qualities=True)

        assert output._qualities is not None
    

    def test_download_features_from_get(self):
        output = self.dataset.get(2,download_features_meta_data=True)

        assert output._features is not None


    def test_get_features(self):
        output = self.dataset.get_features(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 37


    def test_get_qualities(self):
        output = self.dataset.get_qualities(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 19


    def test_status_update(self):
        dataset = OpenMLDataset(
            f"TEST-{str(time.time())}-UploadTestWithURL",
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


    def test_edit(self):
        did = 2
        result = self.dataset.fork(did)
        self._wait_for_dataset_being_processed([did,result])

        edited_did = self.dataset.edit(result,description="Forked dataset", default_target_attribute="shape")
        assert result == edited_did
        n_tries = 10
        # we need to wait for the edit to be reflected on the server
        for i in range(n_tries):
            edited_dataset = self.dataset.get(result,force_refresh_cache=True)
            try:
                assert edited_dataset.default_target_attribute == "shape", edited_dataset
                assert edited_dataset.description == "Forked dataset", edited_dataset
                break
            except AssertionError as e:
                if i == n_tries - 1:
                    raise e
                time.sleep(10)


    def test_fork(self):
        did = 2
        result = self.dataset.fork(did)
        assert did != result
        # wait for processing
        self._wait_for_dataset_being_processed([did,result])

        listing = self.dataset.list(limit=2,offset=0,data_id=[did,result])
        assert listing.iloc[0]["name"] == listing.iloc[1]["name"]

        self.dataset.delete(result)


    def test_list_qualities(self):
        output = self.dataset.list_qualities()
        assert len(output) == 19
        assert isinstance(output[0],str)


    def test_feature_add_remove_ontology(self):
        did = 11
        fid = 0
        ontology = "https://www.openml.org/unittest/" + str(time.time())
        output = self.dataset.feature_add_ontology(did,fid,ontology)
        assert output

        output = self.dataset.feature_remove_ontology(did,fid,ontology)
        assert output


    def test_add_delete_topic(self):
        topic = f"test_topic_{str(time.time())}"
        # only admin can add or delete topics
        self.api_key = self.dataset._http.api_key = self.admin_key

        self.dataset.add_topic(31, topic)
        self.dataset.delete_topic(31, topic)

    
    def _wait_for_dataset_being_processed(self,dids,n_tries=10,wait_time=10):
        for i in range(n_tries):
            time.sleep(wait_time)
            listing = self.dataset.list(limit=2,offset=0,data_id=dids)
            if listing.shape[0] == 2:
                return
        raise TimeoutError("Dataset did not become active within given time")


@pytest.mark.uses_test_server()
class TestDatasetV2API(TestAPIBase):
    def setUp(self):
        super().setUp()
        http_client_v2 = self.http_clients[APIVersion.V2]
        self.dataset = DatasetV2API(http_client_v2,self.minio_client)


    def test_get(self):
        output= self.dataset.get(2)
        assert output.dataset_id == 2
    

    def test_list(self):
        output =self.dataset.list(limit=2, offset=0, status="active")
        assert not output.empty
        assert output.shape[0] == 2
        assert (output["status"].nunique() == 1)
        assert (output["status"].unique()[0] == "active")
    

    def test_download_arff_from_get(self):
        output = self.dataset.get(2,download_data=True)
        
        assert output.data_file != None
        assert Path(output.data_file).exists()


    def test_download_qualities_from_get(self):
        output = self.dataset.get(2,download_qualities=True)

        assert output._qualities is not None
    

    def test_download_features_from_get(self):
        output = self.dataset.get(2,download_features_meta_data=True)

        assert output._features is not None


    def test_get_features(self):
        output = self.dataset.get_features(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 37


    def test_get_qualities(self):
        output = self.dataset.get_qualities(2)

        assert isinstance(output,dict)
        assert len(output.keys()) == 107


    def test_list_qualities(self):
        output = self.dataset.list_qualities()
        assert len(output) == 107
        assert isinstance(output[0],str)


@pytest.mark.uses_test_server()
class TestDatasetsCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        http_client_v1 = self.http_clients[APIVersion.V1]
        http_client_v2 = self.http_clients[APIVersion.V2]
        self.dataset_v1 = DatasetV1API(http_client_v1,self.minio_client)
        self.dataset_v2 = DatasetV2API(http_client_v2,self.minio_client)
        self.dataset_fallback = FallbackProxy(self.dataset_v1,self.dataset_v2)
    

    def test_get_matches(self):
        output_v1 = self.dataset_v1.get(2)
        output_v2 = self.dataset_v2.get(2)

        assert output_v1.dataset_id == output_v2.dataset_id
        assert output_v1.name == output_v2.name
        assert output_v1.data_file is None
        assert output_v1.data_file == output_v2.data_file
    

    def test_get_fallback(self):
        output_fallback =  self.dataset_fallback.get(2)
        assert output_fallback.dataset_id == 2

    
    def test_list_matches(self):
        output_v1 = self.dataset_v1.list(limit=2, offset=1)
        output_v2 = self.dataset_v2.list(limit=2, offset=1)

        pd.testing.assert_frame_equal(
            output_v1[["did","name","version"]],
            output_v2[["did","name","version"]],
            check_like=True
            )

    def test_list_fallback(self):
        output_fallback =self.dataset_fallback.list(limit=2, offset=0,data_id=[2,3])

        assert not output_fallback.empty
        assert output_fallback.shape[0] == 2
        assert set(output_fallback["did"]) == {2, 3}


    def test_get_features_matches(self):
        output_v1 = self.dataset_v1.get_features(3)
        output_v2 = self.dataset_v2.get_features(3)

        assert output_v1.keys() == output_v2.keys()
        # would not be same if v1 has ontology
        assert output_v1 == output_v2


    def test_get_features_fallback(self):
        output_fallback = self.dataset_fallback.get_features(2)

        assert isinstance(output_fallback,dict)
        assert len(output_fallback.keys()) == 37


    def test_get_qualities_matches(self):
        #TODO Qualities in local python server and test server differ
        output_v1 = self.dataset_v1.get_qualities(2)
        output_v2 = self.dataset_v2.get_qualities(2)
        assert output_v1['AutoCorrelation'] == output_v2['AutoCorrelation']


    def test_get_qualities_fallback(self):
        #TODO Qualities in local python server and test server differ
        output_fallback = self.dataset_fallback.get_qualities(2)

        assert isinstance(output_fallback,dict)


    def test_list_qualities_matches(self):
        #TODO Qualities in local python server and test server differ
        output_v1 = self.dataset_v1.list_qualities()
        output_v2 = self.dataset_v2.list_qualities()

        assert "AutoCorrelation" in output_v1
        assert "AutoCorrelation" in output_v2


    def test_list_qualities_fallback(self):
        #TODO Qualities in local python server and test server differ
        output_fallback = self.dataset_fallback.list_qualities()

        assert isinstance(output_fallback,list)


    def test_status_update_fallback(self):
        dataset = OpenMLDataset(
            f"TEST-{str(time.time())}-UploadTestWithURL",
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


    def test_edit_fallback(self):
        did = 2
        result = self.dataset_fallback.fork(did)
        self._wait_for_dataset_being_processed([did,result])

        edited_did = self.dataset_fallback.edit(result,description="Forked dataset", default_target_attribute="shape")
        assert result == edited_did
        n_tries = 10
        # we need to wait for the edit to be reflected on the server
        for i in range(n_tries):
            edited_dataset = self.dataset_fallback.get(result,force_refresh_cache=True)
            try:
                assert edited_dataset.default_target_attribute == "shape", edited_dataset
                assert edited_dataset.description == "Forked dataset", edited_dataset
                break
            except AssertionError as e:
                if i == n_tries - 1:
                    raise e
                time.sleep(10)


    def test_fork_fallback(self):
        did = 2
        result = self.dataset_fallback.fork(did)
        assert did != result
        self._wait_for_dataset_being_processed([did,result])
        
        self.dataset_fallback.delete(result)


    def test_feature_add_remove_ontology_fallback(self):
        ontology = "https://www.openml.org/unittest/" + str(time.time())
        output_fallback_add = self.dataset_fallback.feature_add_ontology(
            11, 0,ontology)
        assert output_fallback_add
        output_fallback_remove = self.dataset_fallback.feature_remove_ontology(
            11, 0,ontology)
        assert output_fallback_remove
    

    def test_add_delete_topic_fallback(self):
        topic = f"test_topic_{str(time.time())}"
        # only admin can add or delete topics
        self.api_key = self.dataset_fallback._http.api_key = self.admin_key
        self.dataset_fallback.add_topic(31, topic)
        self.dataset_fallback.delete_topic(31, topic)


    def _wait_for_dataset_being_processed(self,dids,n_tries=10,wait_time=10):
        for i in range(n_tries):
            time.sleep(wait_time)
            listing = self.dataset_fallback.list(limit=2,offset=0,data_id=dids)
            if listing.shape[0] == 2:
                return
        raise TimeoutError("Dataset did not become active within given time")