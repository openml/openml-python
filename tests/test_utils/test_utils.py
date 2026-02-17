from __future__ import annotations

import os
import unittest.mock
import pytest
import openml
from openml.testing import _check_dataset


@pytest.fixture()
def min_number_tasks_on_test_server() -> int:
    """After a reset at least 1068 tasks are on the test server"""
    return 1068


@pytest.fixture()
def min_number_datasets_on_test_server() -> int:
    """After a reset at least 127 datasets are on the test server"""
    return 127


@pytest.fixture()
def min_number_flows_on_test_server() -> int:
    """After a reset at least 127 flows are on the test server"""
    return 15


@pytest.fixture()
def min_number_setups_on_test_server() -> int:
    """After a reset at least 20 setups are on the test server"""
    return 50


@pytest.fixture()
def min_number_runs_on_test_server() -> int:
    """After a reset at least 21 runs are on the test server"""
    return 21


@pytest.fixture()
def min_number_evaluations_on_test_server() -> int:
    """After a reset at least 8 evaluations are on the test server"""
    return 8


def _mocked_perform_api_call(call, request_method):
    url = openml.config.server + "/" + call
    return openml._api_calls._download_text_file(url)


@pytest.mark.uses_test_server()
def test_list_all():
    openml.utils._list_all(listing_call=openml.tasks.functions._list_tasks)


@pytest.mark.uses_test_server()
def test_list_all_for_tasks(min_number_tasks_on_test_server):
    tasks = openml.tasks.list_tasks(size=min_number_tasks_on_test_server)
    assert min_number_tasks_on_test_server == len(tasks)


@pytest.mark.uses_test_server()
def test_list_all_with_multiple_batches(min_number_tasks_on_test_server):
    # By setting the batch size one lower than the minimum we guarantee at least two
    # batches and at the same time do as few batches (roundtrips) as possible.
    batch_size = min_number_tasks_on_test_server - 1
    batches = openml.utils._list_all(
        listing_call=openml.tasks.functions._list_tasks,
        batch_size=batch_size,
    )
    assert len(batches) >= 2
    assert min_number_tasks_on_test_server <= sum(len(batch) for batch in batches)


@pytest.mark.uses_test_server()
def test_list_all_for_datasets(min_number_datasets_on_test_server):
    datasets = openml.datasets.list_datasets(
        size=min_number_datasets_on_test_server,
    )

    assert min_number_datasets_on_test_server == len(datasets)
    for dataset in datasets.to_dict(orient="index").values():
        _check_dataset(dataset)


@pytest.mark.uses_test_server()
def test_list_all_for_flows(min_number_flows_on_test_server):
    flows = openml.flows.list_flows(size=min_number_flows_on_test_server)
    assert min_number_flows_on_test_server == len(flows)


@pytest.mark.flaky()  # Other tests might need to upload runs first
@pytest.mark.uses_test_server()
def test_list_all_for_setups(min_number_setups_on_test_server):
    # TODO apparently list_setups function does not support kwargs
    setups = openml.setups.list_setups(size=min_number_setups_on_test_server)
    assert min_number_setups_on_test_server == len(setups)


@pytest.mark.flaky()  # Other tests might need to upload runs first
@pytest.mark.uses_test_server()
def test_list_all_for_runs(min_number_runs_on_test_server):
    runs = openml.runs.list_runs(size=min_number_runs_on_test_server)
    assert min_number_runs_on_test_server == len(runs)


@pytest.mark.flaky()  # Other tests might need to upload runs first
@pytest.mark.uses_test_server()
def test_list_all_for_evaluations(min_number_evaluations_on_test_server):
    # TODO apparently list_evaluations function does not support kwargs
    evaluations = openml.evaluations.list_evaluations(
        function="predictive_accuracy",
        size=min_number_evaluations_on_test_server,
    )
    assert min_number_evaluations_on_test_server == len(evaluations)


@unittest.mock.patch("openml._api_calls._perform_api_call", side_effect=_mocked_perform_api_call)
@pytest.mark.uses_test_server()
def test_list_all_few_results_available(_perform_api_call):
    datasets = openml.datasets.list_datasets(size=1000, data_name="iris", data_version=1)
    assert len(datasets) == 1, "only one iris dataset version 1 should be present"
    assert _perform_api_call.call_count == 1, "expect just one call to get one dataset"


@unittest.skipIf(os.name == "nt", "https://github.com/openml/openml-python/issues/1033")
@unittest.mock.patch("openml.config.get_cache_directory")
def test__create_cache_directory(config_mock, tmp_path):
    config_mock.return_value = tmp_path
    openml.utils._create_cache_directory("abc")
    assert (tmp_path / "abc").exists()

    subdir = tmp_path / "def"
    subdir.mkdir()
    subdir.chmod(0o444)
    config_mock.return_value = subdir
    with pytest.raises(
        openml.exceptions.OpenMLCacheException,
        match="Cannot create cache directory",
    ):
        openml.utils._create_cache_directory("ghi")


@pytest.mark.uses_test_server()
def test_correct_test_server_download_state():
    """This test verifies that the test server downloads the data from the correct source.

    If this tests fails, it is highly likely that the test server is not configured correctly.
    Usually, this means that the test server is serving data from the task with the same ID from the production server.
    That is, it serves parquet files wrongly associated with the test server's task.
    """
    task = openml.tasks.get_task(119)
    dataset = task.get_dataset()
    assert len(dataset.features) == dataset.get_data()[0].shape[1]

@unittest.mock.patch("openml.config.get_cache_directory")
def test_get_cache_size(config_mock,tmp_path):
    """
    Test that the OpenML cache size utility correctly reports the cache directory
    size before and after fetching a dataset.

    This test uses a temporary directory (tmp_path) as the cache location by
    patching the configuration via config_mock. It verifies two conditions:
    empty cache and after dataset fetch. 

    Parameters
    ----------
    config_mock : unittest.mock.Mock
         A mock that overrides the configured cache directory to point to tmp_path.
    tmp_path : pathlib.Path
         A pytest-provided temporary directory used as an isolated cache location.
    """
    
    config_mock.return_value = tmp_path
    cache_size = openml.utils.get_cache_size()
    assert cache_size == 0
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    (sub_dir / "nested_file.txt").write_bytes(b"b" * 100)
    
    assert openml.utils.get_cache_size() == 100