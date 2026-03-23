from __future__ import annotations
from pathlib import Path
import time
import os

from openml import OpenMLDataset
import pytest
import pandas as pd

import openml
from openml.testing import TestBase
from openml.exceptions import OpenMLNotSupportedError
from openml._api import DatasetV1API, DatasetV2API

@pytest.fixture
def dataset_v1(http_client_v1, minio_client) -> DatasetV1API:
    return DatasetV1API(http=http_client_v1, minio=minio_client)

@pytest.fixture
def dataset_v2(http_client_v2, minio_client) -> DatasetV2API:
    return DatasetV2API(http=http_client_v2, minio=minio_client)


def _wait_for_dataset_being_processed(dataset, did, status='active', n_tries=10, wait_time=10):
    for _ in range(n_tries):
        try:
            time.sleep(wait_time)
            result = dataset.list(limit=1, offset=0, data_id=[did], status="all")
            result = result.to_dict(orient="index")
            if result[did]["status"] == status:
                return
        except Exception:
            pass
    raise TimeoutError(f"Dataset did not become {status} within given time")

def _status_update_check(dataset, dataset_id, status):
    dataset.status_update(dataset_id, status)
    _wait_for_dataset_being_processed(dataset, dataset_id, status)


@pytest.mark.test_server()
def test_v1_get(dataset_v1):
    dataset_id = 2
    output = dataset_v1.get(dataset_id)
    assert output.dataset_id == dataset_id

@pytest.mark.test_server()
def test_v1_list(dataset_v1):
    output = dataset_v1.list(limit=2, offset=0, status="active")
    assert not output.empty
    assert output.shape[0] == 2
    assert output["status"].nunique() == 1
    assert output["status"].unique()[0] == "active"

@pytest.mark.test_server()
def test_v1_download_arff(dataset_v1):
    from openml.datasets.functions import _get_dataset_arff
    output = dataset_v1.get(2)
    file = _get_dataset_arff(output)
    assert file.exists()

@pytest.mark.test_server()
def test_v1_download_parquet(dataset_v1):
    from openml.datasets.functions import _get_dataset_parquet
    output = dataset_v1.get(2)
    file = _get_dataset_parquet(output)
    assert file.exists()

@pytest.mark.test_server()
def test_v1_download_arff_from_get(dataset_v1):
    output = dataset_v1.get(2, download_data=True)
    data = output.data_file is not None and Path(output.data_file).exists()
    parquet = output.parquet_file is not None and Path(output.parquet_file).exists()
    assert data or parquet

@pytest.mark.test_server()
def test_v1_download_qualities_from_get(dataset_v1):
    output = dataset_v1.get(2, download_qualities=True)

    assert output._qualities is not None

@pytest.mark.test_server()
def test_v1_download_features_from_get(dataset_v1):
    output = dataset_v1.get(2, download_features_meta_data=True)

    assert output._features is not None

@pytest.mark.test_server()
def test_v1_get_features(dataset_v1):
    output = dataset_v1.get_features(2)

    assert isinstance(output, dict)
    assert len(output.keys()) == 37

@pytest.mark.test_server()
def test_v1_get_qualities(dataset_v1):
    output = dataset_v1.get_qualities(2)

    assert isinstance(output, dict)
    assert len(output.keys()) == 107

@pytest.mark.skipif(
    not os.environ.get(openml.config.OPENML_TEST_SERVER_ADMIN_KEY_ENV_VAR),
    reason="Test requires admin key. Set OPENML_TEST_SERVER_ADMIN_KEY environment variable.",
)
@pytest.mark.test_server()
def test_v1_status_update(dataset_v1):
    openml.config.apikey = TestBase.admin_key
    new_dataset = OpenMLDataset(
        f"TEST-{str(time.time())}-UploadTestWithURL",
        "test",
        "ARFF",
        version=1,
        url="https://www.openml.org/data/download/61/dataset_61_iris.arff",
    )
    new_dataset.publish()
    _status_update_check(dataset_v1, new_dataset.dataset_id, "deactivated")
    _status_update_check(dataset_v1, new_dataset.dataset_id, "active")
    dataset_v1.delete(new_dataset.dataset_id)

@pytest.mark.test_server()
def test_v1_edit(dataset_v1):
    did = 2
    result = dataset_v1.fork(did)
    _wait_for_dataset_being_processed(dataset_v1, result,'in_preparation')

    edited_did = dataset_v1.edit(result, description="Forked dataset", default_target_attribute="shape")
    assert result == edited_did
    n_tries = 10
    # we need to wait for the edit to be reflected on the server
    for i in range(n_tries):
        edited_dataset = dataset_v1.get(result, force_refresh_cache=True)
        try:
            assert edited_dataset.default_target_attribute == "shape", edited_dataset
            assert edited_dataset.description == "Forked dataset", edited_dataset
            break
        except AssertionError as e:
            if i == n_tries - 1:
                raise e
            time.sleep(10)

@pytest.mark.test_server()
def test_v1_fork(dataset_v1):
    did = 2
    result = dataset_v1.fork(did)
    assert did != result
    _wait_for_dataset_being_processed(dataset_v1, result,'in_preparation')

    listing = dataset_v1.list(limit=2, offset=0, data_id=[did, result], status="all")

    assert listing.iloc[0]["name"] == listing.iloc[1]["name"]
    dataset_v1.delete(result)

@pytest.mark.test_server()
def test_v1_list_qualities(dataset_v1):
    output = dataset_v1.list_qualities()
    assert len(output) == 107
    assert isinstance(output[0], str)

@pytest.mark.test_server()
def test_v1_feature_add_remove_ontology(dataset_v1):
    did = 11
    fid = 0
    ontology = "https://www.openml.org/unittest/" + str(time.time())
    output = dataset_v1.feature_add_ontology(did, fid, ontology)
    assert output

    output = dataset_v1.feature_remove_ontology(did, fid, ontology)
    assert output

@pytest.mark.skipif(
    not os.environ.get(openml.config.OPENML_TEST_SERVER_ADMIN_KEY_ENV_VAR),
    reason="Test requires admin key. Set OPENML_TEST_SERVER_ADMIN_KEY environment variable.",
)
@pytest.mark.test_server()
def test_v1_add_delete_topic(dataset_v1):
    openml.config.apikey = TestBase.admin_key
    topic = f"test_topic_{str(time.time())}"
    dataset_v1.add_topic(31, topic)
    dataset_v1.delete_topic(31, topic)

@pytest.mark.test_server()
def test_v2_get(dataset_v2):
    dataset_id = 2
    output = dataset_v2.get(dataset_id)
    assert output.dataset_id == dataset_id

@pytest.mark.test_server()
def test_v2_list(dataset_v2):
    output = dataset_v2.list(limit=2, offset=0, status="active")
    assert not output.empty
    assert output.shape[0] == 2
    assert output["status"].nunique() == 1
    assert output["status"].unique()[0] == "active"

@pytest.mark.test_server()
def test_v2_download_arff(dataset_v2):
    from openml.datasets.functions import _get_dataset_arff
    output = dataset_v2.get(2)
    file = _get_dataset_arff(output)
    assert file.exists()

@pytest.mark.test_server()
def test_v2_download_parquet(dataset_v2):
    from openml.datasets.functions import _get_dataset_parquet
    output = dataset_v2.get(2)
    file = _get_dataset_parquet(output)
    assert file.exists()

@pytest.mark.test_server()
def test_v2_download_arff_from_get(dataset_v2):
    output = dataset_v2.get(2, download_data=True)
    data = output.data_file is not None and Path(output.data_file).exists()
    parquet = output.parquet_file is not None and Path(output.parquet_file).exists()
    assert data or parquet

@pytest.mark.test_server()
def test_v2_download_qualities_from_get(dataset_v2):
    output = dataset_v2.get(2, download_qualities=True)

    assert output._qualities is not None

@pytest.mark.test_server()
def test_v2_download_features_from_get(dataset_v2):
    output = dataset_v2.get(2, download_features_meta_data=True)

    assert output._features is not None

@pytest.mark.test_server()
def test_v2_get_features(dataset_v2):
    output = dataset_v2.get_features(2)

    assert isinstance(output, dict)
    assert len(output.keys()) == 37

@pytest.mark.test_server()
def test_v2_edit(dataset_v2):
    with pytest.raises(OpenMLNotSupportedError):
        dataset_v2.edit(2, description='Test')

@pytest.mark.test_server()
def test_v2_fork(dataset_v2):
    with pytest.raises(OpenMLNotSupportedError):
        dataset_v2.fork(2)

@pytest.mark.test_server()
def test_v2_feature_add_remove_ontology(dataset_v2):
    with pytest.raises(OpenMLNotSupportedError):
        dataset_v2.feature_add_ontology(2, 0, "https://www.openml.org/unittest/" + str(time.time()))

@pytest.mark.test_server()
def test_v2_add_delete_topic(dataset_v2):
    with pytest.raises(OpenMLNotSupportedError):
        dataset_v2.add_topic(2, 'test_topic_' + str(time.time()))

@pytest.mark.test_server()
def test_v2_get_qualities(dataset_v2):
    output = dataset_v2.get_qualities(2)
    assert isinstance(output, dict)
    assert len(output.keys()) == 107

@pytest.mark.test_server()
def test_v2_list_qualities(dataset_v2):
    output = dataset_v2.list_qualities()
    assert len(output) == 107
    assert isinstance(output[0], str)

@pytest.mark.skip(reason="Needs valid v2 admin key required")
@pytest.mark.test_server()
def test_v2_status_update(dataset_v2):
    openml.config.apikey = TestBase.admin_key
    # publish and fork is not supported in v2
    _status_update_check(dataset_v2, 2, "deactivated")
    _status_update_check(dataset_v2, 2, "active")

@pytest.mark.test_server()
def test_get_matches(dataset_v1, dataset_v2):
    output_v1 = dataset_v1.get(2)
    output_v2 = dataset_v2.get(2)

    assert output_v1.dataset_id == output_v2.dataset_id
    assert output_v1.name == output_v2.name
    assert output_v1.data_file is None
    assert output_v1.data_file == output_v2.data_file

@pytest.mark.test_server()
def test_get_features_matches(dataset_v1, dataset_v2):
    output_v1 = dataset_v1.get_features(3)
    output_v2 = dataset_v2.get_features(3)

    assert output_v1.keys() == output_v2.keys()
    # would not be same if v1 has ontology
    assert output_v1 == output_v2

@pytest.mark.test_server()
def test_list_matches(dataset_v1, dataset_v2):
    output_v1 = dataset_v1.list(limit=2, offset=1)
    output_v2 = dataset_v2.list(limit=2, offset=1)

    pd.testing.assert_frame_equal(
        output_v1[["did", "name", "version"]],
        output_v2[["did", "name", "version"]],
        check_like=True
        )
    
@pytest.mark.test_server()
def test_get_qualities_matches(dataset_v1, dataset_v2):
    output_v1 = dataset_v1.get_qualities(2)
    output_v2 = dataset_v2.get_qualities(2)
    assert  output_v1['AutoCorrelation'] == output_v2['AutoCorrelation']
    assert len(output_v1) == len(output_v2)

@pytest.mark.test_server()
def test_list_qualities_matches(dataset_v1, dataset_v2):
    output_v1 = dataset_v1.list_qualities()
    output_v2 = dataset_v2.list_qualities()

    assert output_v1 == output_v2

