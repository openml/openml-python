from __future__ import annotations

import os
import unittest.mock
import pytest
import shutil
import openml
from openml.testing import _check_dataset


@pytest.fixture(autouse=True)
def as_robot():
    policy = openml.config.retry_policy
    n_retries = openml.config.connection_n_retries
    openml.config.set_retry_policy("robot", n_retries=20)
    yield
    openml.config.set_retry_policy(policy, n_retries)


@pytest.fixture(autouse=True)
def with_test_server():
    openml.config.start_using_configuration_for_example()
    yield
    openml.config.stop_using_configuration_for_example()


@pytest.fixture(autouse=True)
def with_test_cache(test_files_directory, request):
    if not test_files_directory.exists():
        raise ValueError(
            f"Cannot find test cache dir, expected it to be {test_files_directory!s}!",
        )
    _root_cache_directory = openml.config._root_cache_directory
    tmp_cache = test_files_directory / request.node.name
    openml.config.set_root_cache_directory(tmp_cache)
    yield
    openml.config.set_root_cache_directory(_root_cache_directory)
    if tmp_cache.exists():
        shutil.rmtree(tmp_cache)


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
    """After a reset at least 50 setups are on the test server"""
    return 50


@pytest.fixture()
def min_number_runs_on_test_server() -> int:
    """After a reset at least 50 runs are on the test server"""
    return 21


@pytest.fixture()
def min_number_evaluations_on_test_server() -> int:
    """After a reset at least 22 evaluations are on the test server"""
    return 22


def _mocked_perform_api_call(call, request_method):
    url = openml.config.server + "/" + call
    return openml._api_calls._download_text_file(url)


@pytest.mark.server()
def test_list_all():
    openml.utils._list_all(listing_call=openml.tasks.functions._list_tasks)
    openml.utils._list_all(
        listing_call=openml.tasks.functions._list_tasks,
        list_output_format="dataframe",
    )


@pytest.mark.server()
def test_list_all_for_tasks(min_number_tasks_on_test_server):
    tasks = openml.tasks.list_tasks(
        batch_size=1000,
        size=min_number_tasks_on_test_server,
        output_format="dataframe",
    )
    assert min_number_tasks_on_test_server == len(tasks)


@pytest.mark.server()
def test_list_all_with_multiple_batches(min_number_tasks_on_test_server):
    # By setting the batch size one lower than the minimum we guarantee at least two
    # batches and at the same time do as few batches (roundtrips) as possible.
    batch_size = min_number_tasks_on_test_server - 1
    res = openml.utils._list_all(
        listing_call=openml.tasks.functions._list_tasks,
        list_output_format="dataframe",
        batch_size=batch_size,
    )
    assert min_number_tasks_on_test_server <= len(res)


@pytest.mark.server()
def test_list_all_for_datasets(min_number_datasets_on_test_server):
    datasets = openml.datasets.list_datasets(
        batch_size=100,
        size=min_number_datasets_on_test_server,
        output_format="dataframe",
    )

    assert min_number_datasets_on_test_server == len(datasets)
    for dataset in datasets.to_dict(orient="index").values():
        _check_dataset(dataset)


@pytest.mark.server()
def test_list_all_for_flows(min_number_flows_on_test_server):
    flows = openml.flows.list_flows(
        batch_size=25,
        size=min_number_flows_on_test_server,
        output_format="dataframe",
    )
    assert min_number_flows_on_test_server == len(flows)


@pytest.mark.server()
@pytest.mark.flaky()  # Other tests might need to upload runs first
def test_list_all_for_setups(min_number_setups_on_test_server):
    # TODO apparently list_setups function does not support kwargs
    setups = openml.setups.list_setups(size=min_number_setups_on_test_server)
    assert min_number_setups_on_test_server == len(setups)


@pytest.mark.server()
@pytest.mark.flaky()  # Other tests might need to upload runs first
def test_list_all_for_runs(min_number_runs_on_test_server):
    runs = openml.runs.list_runs(batch_size=25, size=min_number_runs_on_test_server)
    assert min_number_runs_on_test_server == len(runs)


@pytest.mark.server()
@pytest.mark.flaky()  # Other tests might need to upload runs first
def test_list_all_for_evaluations(min_number_evaluations_on_test_server):
    # TODO apparently list_evaluations function does not support kwargs
    evaluations = openml.evaluations.list_evaluations(
        function="predictive_accuracy",
        size=min_number_evaluations_on_test_server,
    )
    assert min_number_evaluations_on_test_server == len(evaluations)


@pytest.mark.server()
@unittest.mock.patch("openml._api_calls._perform_api_call", side_effect=_mocked_perform_api_call)
def test_list_all_few_results_available(_perform_api_call):
    datasets = openml.datasets.list_datasets(
        size=1000,
        data_name="iris",
        data_version=1,
        output_format="dataframe",
    )
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


@pytest.mark.server()
def test_correct_test_server_download_state():
    """This test verifies that the test server downloads the data from the correct source.

    If this tests fails, it is highly likely that the test server is not configured correctly.
    Usually, this means that the test server is serving data from the task with the same ID from the production server.
    That is, it serves parquet files wrongly associated with the test server's task.
    """
    task = openml.tasks.get_task(119)
    dataset = task.get_dataset()
    assert len(dataset.features) == dataset.get_data(dataset_format="dataframe")[0].shape[1]