from __future__ import annotations

import os
import unittest.mock

import pandas as pd
import pytest

import openml
from openml.evaluations.evaluation import OpenMLEvaluation
from openml.setups.setup import OpenMLSetup
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
    return 15


@pytest.fixture()
def min_number_evaluations_on_test_server() -> int:
    """After a reset at least 8 evaluations are on the test server"""
    return 8



def _create_mock_listing_call(total_items, item_factory, return_type="dataframe"):
    def mock_listing_call(limit, offset, **kwargs):
        if offset >= total_items:
            return pd.DataFrame() if return_type == "dataframe" else []
        size = min(limit, total_items - offset)
        items = [item_factory(i) for i in range(offset, offset + size)]
        return pd.DataFrame(items) if return_type == "dataframe" else items
    return mock_listing_call

def _mocked_perform_api_call(call, request_method):
    if call == "data/list/limit/1000/offset/0/data_name/iris/data_version/1":
        return """<oml:data xmlns:oml="http://openml.org/openml">
            <oml:dataset>
                <oml:did>61</oml:did>
                <oml:name>iris</oml:name>
                <oml:version>1</oml:version>
                <oml:status>active</oml:status>
            </oml:dataset>
        </oml:data>"""
    raise ValueError(f"Unexpected call: {call}")


@unittest.mock.patch("openml.tasks.functions._list_tasks")
def test_list_all(mock_list_tasks):
    mock_list_tasks.side_effect = _create_mock_listing_call(10, lambda i: {"tid": i})
    openml.utils._list_all(listing_call=openml.tasks.functions._list_tasks)


@unittest.mock.patch("openml.tasks.functions._list_tasks")
def test_list_all_for_tasks(mock_list_tasks, min_number_tasks_on_test_server):
    mock_list_tasks.side_effect = _create_mock_listing_call(
        min_number_tasks_on_test_server, lambda i: {"tid": i}
    )
    tasks = openml.tasks.list_tasks(size=min_number_tasks_on_test_server)
    assert min_number_tasks_on_test_server == len(tasks)


@unittest.mock.patch("openml.tasks.functions._list_tasks")
def test_list_all_with_multiple_batches(mock_list_tasks, min_number_tasks_on_test_server):
    mock_list_tasks.side_effect = _create_mock_listing_call(
        min_number_tasks_on_test_server, lambda i: {"tid": i}
    )
    # By setting the batch size one lower than the minimum we guarantee at least two
    # batches and at the same time do as few batches (roundtrips) as possible.
    batch_size = min_number_tasks_on_test_server - 1
    batches = openml.utils._list_all(
        listing_call=openml.tasks.functions._list_tasks,
        batch_size=batch_size,
    )
    assert len(batches) >= 2
    assert min_number_tasks_on_test_server <= sum(len(batch) for batch in batches)


@unittest.mock.patch("openml.datasets.functions._list_datasets")
def test_list_all_for_datasets(mock_list_datasets, min_number_datasets_on_test_server):
    mock_list_datasets.side_effect = _create_mock_listing_call(
        min_number_datasets_on_test_server, lambda i: {"did": i, "status": "active"}
    )
    datasets = openml.datasets.list_datasets(
        size=min_number_datasets_on_test_server,
    )

    assert min_number_datasets_on_test_server == len(datasets)
    for dataset in datasets.to_dict(orient="index").values():
        _check_dataset(dataset)


@unittest.mock.patch("openml.flows.functions._list_flows")
def test_list_all_for_flows(mock_list_flows, min_number_flows_on_test_server):
    mock_list_flows.side_effect = _create_mock_listing_call(
        min_number_flows_on_test_server, lambda i: {"id": i}
    )
    flows = openml.flows.list_flows(size=min_number_flows_on_test_server)
    assert min_number_flows_on_test_server == len(flows)


@unittest.mock.patch("openml.setups.functions._list_setups")
def test_list_all_for_setups(mock_list_setups, min_number_setups_on_test_server):
    mock_list_setups.side_effect = _create_mock_listing_call(
        min_number_setups_on_test_server,
        lambda i: OpenMLSetup(setup_id=i, flow_id=1, parameters={}),
        return_type="list"
    )
    # TODO apparently list_setups function does not support kwargs
    setups = openml.setups.list_setups(size=min_number_setups_on_test_server)
    assert min_number_setups_on_test_server == len(setups)


@unittest.mock.patch("openml.runs.functions._list_runs")
def test_list_all_for_runs(mock_list_runs, min_number_runs_on_test_server):
    mock_list_runs.side_effect = _create_mock_listing_call(
        min_number_runs_on_test_server, lambda i: {"run_id": i}
    )
    runs = openml.runs.list_runs(size=min_number_runs_on_test_server)
    assert min_number_runs_on_test_server == len(runs)


@unittest.mock.patch("openml.evaluations.functions._list_evaluations")
def test_list_all_for_evaluations(mock_list_evaluations, min_number_evaluations_on_test_server):
    mock_list_evaluations.side_effect = _create_mock_listing_call(
        min_number_evaluations_on_test_server,
        lambda i: OpenMLEvaluation(
            run_id=i, task_id=1, setup_id=1, flow_id=1, flow_name="flow", data_id=1, data_name="data",
            function="predictive_accuracy", upload_time="2020-01-01", uploader=1, uploader_name="user",
            value=0.5, values=None
        ),
        return_type="list"
    )
    # TODO apparently list_evaluations function does not support kwargs
    evaluations = openml.evaluations.list_evaluations(
        function="predictive_accuracy",
        size=min_number_evaluations_on_test_server,
    )
    assert min_number_evaluations_on_test_server == len(evaluations)


@unittest.mock.patch("openml._api_calls._perform_api_call", side_effect=_mocked_perform_api_call)
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


@unittest.mock.patch("openml.tasks.get_task")
def test_correct_test_server_download_state(mock_get_task):
    """This test verifies that the test server downloads the data from the correct source.

    If this tests fails, it is highly likely that the test server is not configured correctly.
    Usually, this means that the test server is serving data from the task with the same ID from the production server.
    That is, it serves parquet files wrongly associated with the test server's task.
    """
    mock_task = unittest.mock.Mock()
    mock_dataset = unittest.mock.Mock()
    mock_dataset.features = {0: "feature1", 1: "feature2"}
    mock_dataset.get_data.return_value = (pd.DataFrame({"feature1": [1], "feature2": [2]}), None, None, None)
    mock_task.get_dataset.return_value = mock_dataset
    mock_get_task.return_value = mock_task

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
