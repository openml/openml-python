from __future__ import annotations
from pathlib import Path
import time

from openml import OpenMLDataset
import pytest
import pandas as pd
from openml._api.resources.base.fallback import FallbackProxy
from openml._api.resources.base.resources import DatasetAPI
from openml.enums import APIVersion
from openml.testing import TestAPIBase, TestBase
from openml._api.resources.dataset import DatasetV1API, DatasetV2API
from openml.exceptions import OpenMLNotSupportedError


@pytest.mark.uses_test_server()
class TestDatasetBase(TestAPIBase):
    dataset: DatasetAPI
    
    def _wait_for_dataset_being_processed(self,did,status='active',n_tries=10,wait_time=10):
        for i in range(n_tries):
            try:
                time.sleep(wait_time)
                result = self.dataset.list(limit=1,offset=0,data_id=[did], status="all")
                result = result.to_dict(orient="index")
                if result[did]["status"]==status:
                    return
            except:pass
        raise TimeoutError(f"Dataset did not become {status} within given time")
    
    def _status_update_check(self,dataset_id,status):
        self.dataset.status_update(dataset_id,status)
        self._wait_for_dataset_being_processed(dataset_id,status)


class TestDatasetV1API(TestDatasetBase):
    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V1]
        self.dataset = DatasetV1API(http_client)

    def test_get(self):
        output = self.dataset.get(2)
        self.assertEqual(output.dataset_id, 2)

    def test_list(self):
        output =self.dataset.list(limit=2, offset=0, status="active")
        self.assertFalse(output.empty)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output["status"].nunique(), 1)
        self.assertEqual(output["status"].unique()[0], "active")

    def test_download_arff_from_get(self):
        output = self.dataset.get(2,download_data=True)

        self.assertIsNotNone(output.data_file)
        self.assertTrue(Path(output.data_file).exists())

    def test_download_qualities_from_get(self):
        output = self.dataset.get(2,download_qualities=True)

        self.assertIsNotNone(output._qualities)

    def test_download_features_from_get(self):
        output = self.dataset.get(2,download_features_meta_data=True)

        self.assertIsNotNone(output._features)

    def test_get_features(self):
        output = self.dataset.get_features(2)

        self.assertIsInstance(output,dict)
        self.assertEqual(len(output.keys()), 37)

    def test_get_qualities(self):
        output = self.dataset.get_qualities(2)

        self.assertIsInstance(output,dict)
        self.assertEqual(len(output.keys()), 19)

    def test_status_update(self):
        dataset = OpenMLDataset(
            f"TEST-{str(time.time())}-UploadTestWithURL",
            "test",
            "ARFF",
            version=1,
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff",
        )
        dataset.publish()
        # admin key for test server (only admins can activate datasets.
        # all users can deactivate their own datasets)
        self.api_key = self.dataset._http.api_key = self.admin_key
        self._status_update_check(dataset.dataset_id,"deactivated")
        self._status_update_check(dataset.dataset_id,"active")
        TestBase._mark_entity_for_removal("data", dataset.dataset_id)

    def test_edit(self):
        did = 2
        result = self.dataset.fork(did)
        self._wait_for_dataset_being_processed(result)

        edited_did = self.dataset.edit(result,description="Forked dataset", default_target_attribute="shape")
        self.assertEqual(result,edited_did)
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
        self.assertNotEqual(did, result)
        self._wait_for_dataset_being_processed(result)

        listing = self.dataset.list(limit=2,offset=0,data_id=[did,result])
        self.assertEqual(listing.iloc[0]["name"], listing.iloc[1]["name"])

        self.dataset.delete(result)

    def test_list_qualities(self):
        output = self.dataset.list_qualities()
        self.assertEqual(len(output), 19)
        self.assertIsInstance(output[0],str)

    def test_feature_add_remove_ontology(self):
        did = 11
        fid = 0
        ontology = "https://www.openml.org/unittest/" + str(time.time())
        output = self.dataset.feature_add_ontology(did,fid,ontology)
        self.assertTrue(output)

        output = self.dataset.feature_remove_ontology(did,fid,ontology)
        self.assertTrue(output)

    def test_add_delete_topic(self):
        topic = f"test_topic_{str(time.time())}"
        # only admin can add or delete topics
        self.api_key = self.dataset._http.api_key = self.admin_key

        self.dataset.add_topic(31, topic)
        self.dataset.delete_topic(31, topic)


class TestDatasetV2API(TestDatasetV1API):
    def setUp(self):
        super().setUp()
        http_client = self.http_clients[APIVersion.V2]
        self.dataset = DatasetV2API(http_client,self.minio_client)
    
    def test_edit(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_edit()

    def test_fork(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_fork()
    
    def test_feature_add_remove_ontology(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_feature_add_remove_ontology()

    def test_add_delete_topic(self):
        with pytest.raises(OpenMLNotSupportedError):
            super().test_add_delete_topic()

    def test_get_qualities(self):
        # can be removed from here once v2 qualities are same as v1
        output = self.dataset.get_qualities(2)
        self.assertIsInstance(output,dict)
        self.assertEqual(len(output.keys()), 107)

    def test_list_qualities(self):
        # can be removed from here once v2 qualities are same as v1
        output = self.dataset.list_qualities()
        self.assertEqual(len(output), 107)
        self.assertIsInstance(output[0],str)

    def test_status_update(self):
        # publish and fork is not supported in v2
        self._status_update_check(2,"deactivated")
        self._status_update_check(2,"active")


    
class TestResourceCombinedAPI(TestDatasetBase):
    def setUp(self):
        super().setUp()
        http_client_v1 = self.http_clients[APIVersion.V1]
        self.dataset_v1 = DatasetV1API(http_client_v1)

        http_client_v2 = self.http_clients[APIVersion.V2]
        self.dataset_v2 = DatasetV2API(http_client_v2,self.minio_client)
    
    def test_get_matches(self):
        output_v1 = self.dataset_v1.get(2)
        output_v2 = self.dataset_v2.get(2)

        self.assertEqual(output_v1.dataset_id, output_v2.dataset_id)
        self.assertEqual(output_v1.name, output_v2.name)
        self.assertIsNone(output_v1.data_file)
        self.assertEqual(output_v1.data_file, output_v2.data_file)

    def test_get_features_matches(self):
        output_v1 = self.dataset_v1.get_features(3)
        output_v2 = self.dataset_v2.get_features(3)

        self.assertEqual(output_v1.keys(), output_v2.keys())
        # would not be same if v1 has ontology
        self.assertEqual(output_v1, output_v2)

    def test_list_matches(self):
        output_v1 = self.dataset_v1.list(limit=2, offset=1)
        output_v2 = self.dataset_v2.list(limit=2, offset=1)

        pd.testing.assert_frame_equal(
            output_v1[["did","name","version"]],
            output_v2[["did","name","version"]],
            check_like=True
            )
    
    def test_get_qualities_matches(self):
        #TODO Qualities in local python server and test server differ
        output_v1 = self.dataset_v1.get_qualities(2)
        output_v2 = self.dataset_v2.get_qualities(2)
        self.assertEqual(output_v1['AutoCorrelation'], output_v2['AutoCorrelation'])

    def test_list_qualities_matches(self):
        #TODO Qualities in local python server and test server differ
        output_v1 = self.dataset_v1.list_qualities()
        output_v2 = self.dataset_v2.list_qualities()

        self.assertIn("AutoCorrelation", output_v1)
        self.assertIn("AutoCorrelation", output_v2)

