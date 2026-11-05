# License: BSD 3-Clause
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import requests

import openml
from openml import OpenMLSplit, OpenMLTask
from openml.exceptions import (
    OpenMLNotAuthorizedError,
    OpenMLServerException,
)
from openml.tasks import TaskType
from openml.testing import TestBase, create_request_response


def _cached_split_path(task_id: int) -> Path:
    return Path(
        openml.config.get_cache_directory(),
        "api_splits",
        "get",
        str(task_id),
        f"Task_{task_id}_splits.arff",
        "body.arff",
    )


class TestTask(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @pytest.mark.test_server
    def test__get_estimation_procedure_list(self):
        estimation_procedures = openml.tasks.functions._get_estimation_procedure_list()
        assert isinstance(estimation_procedures, list)
        assert isinstance(estimation_procedures[0], dict)
        assert estimation_procedures[0]["task_type_id"] == TaskType.SUPERVISED_CLASSIFICATION

    # 2026-10-20: Removed skip marker, test just works (#1544)
    @pytest.mark.production_server
    def test_list_clustering_task(self):
        self.use_production_server()
        # as shown by #383, clustering tasks can give list/dict casting problems
        openml.tasks.list_tasks(task_type=TaskType.CLUSTERING, size=10)
        # the expected outcome is that it doesn't crash. No assertions.

    def _check_task(self, task):
        assert type(task) == dict  # noqa: E721
        assert len(task) >= 2
        assert "did" in task
        assert isinstance(task["did"], int)
        assert "status" in task
        assert isinstance(task["status"], str)
        assert task["status"] in ["in_preparation", "active", "deactivated"]

    @pytest.mark.test_server
    def test_list_tasks_by_type(self):
        num_curves_tasks = 198  # number is flexible, check server if fails
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid)
        assert len(tasks) >= num_curves_tasks
        for task in tasks.to_dict(orient="index").values():
            assert ttid == task["ttid"]
            self._check_task(task)

    @pytest.mark.test_server
    def test_list_tasks_length(self):
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid)
        assert len(tasks) > 100

    @pytest.mark.test_server
    def test_list_tasks_empty(self):
        tasks = openml.tasks.list_tasks(tag="NoOneWillEverUseThisTag")
        assert tasks.empty

    @pytest.mark.test_server
    def test_list_tasks_by_tag(self):
        # Server starts with 99 active tasks with the tag, and one 'in_preparation',
        # so depending on the processing of the last dataset, there may be 99 or 100 matches.
        num_basic_tasks = 99
        tasks = openml.tasks.list_tasks(tag="OpenML100")
        assert len(tasks) >= num_basic_tasks
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)

    @pytest.mark.test_server
    def test_list_tasks(self):
        tasks = openml.tasks.list_tasks()
        assert len(tasks) >= 900
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)

    @pytest.mark.test_server
    def test_list_tasks_paginate(self):
        size = 10
        max = 100  # noqa: A001
        for i in range(0, max, size):
            tasks = openml.tasks.list_tasks(offset=i, size=size)
            assert size >= len(tasks)
            for task in tasks.to_dict(orient="index").values():
                self._check_task(task)

    @pytest.mark.test_server
    def test_list_tasks_per_type_paginate(self):
        size = 40
        max = 100  # noqa: A001
        task_types = [
            TaskType.SUPERVISED_CLASSIFICATION,
            TaskType.SUPERVISED_REGRESSION,
            TaskType.LEARNING_CURVE,
        ]
        for j in task_types:
            for i in range(0, max, size):
                tasks = openml.tasks.list_tasks(task_type=j, offset=i, size=size)
                assert size >= len(tasks)
                for task in tasks.to_dict(orient="index").values():
                    assert j == task["ttid"]
                    self._check_task(task)

    @pytest.mark.test_server
    def test__get_task(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        with mock.patch("requests.sessions.Session.request") as mock_request:
            openml.tasks.get_task(1882)
            mock_request.assert_not_called()

    @pytest.mark.skip(
        reason="Please await outcome of discussion: https://github.com/openml/OpenML/issues/776",
    )
    @pytest.mark.production_server
    def test__get_task_live(self):
        self.use_production_server()
        # Test the following task as it used to throw an Unicode Error.
        # https://github.com/openml/openml-python/issues/378
        openml.tasks.get_task(34536)

    @pytest.mark.test_server
    def test_get_task(self):
        task = openml.tasks.get_task(1, download_data=True)  # anneal; crossvalidation
        dataset = openml.datasets.get_dataset(task.dataset_id)
        assert isinstance(task, OpenMLTask)
        assert os.path.exists(  # noqa: PTH110
            openml.config.get_minio_download_path(dataset._parquet_url)
        )

    @pytest.mark.test_server
    def test_get_task_lazy(self):
        task = openml.tasks.get_task(2, download_data=False)  # anneal; crossvalidation

        assert isinstance(task, OpenMLTask)
        assert task.class_labels == ["1", "2", "3", "4", "5", "U"]

        # Since the download_data=False is propagated to get_dataset
        assert not os.path.exists(  # noqa: PTH110
            os.path.join(openml.config.get_cache_directory(), "datasets", "2", "dataset.arff")  # noqa: PTH118
        )

        assert not _cached_split_path(2).exists()
        task.download_split()
        assert _cached_split_path(2).exists()

    @pytest.mark.test_server
    def test_get_task_with_cache(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        with (
            mock.patch("openml.tasks.functions.get_dataset"),
            mock.patch("requests.sessions.Session.request") as mock_request,
        ):
            task = openml.tasks.get_task(1)
            mock_request.assert_not_called()

        assert isinstance(task, OpenMLTask)

    @pytest.mark.production_server
    def test_get_task_different_types(self):
        self.use_production_server()
        # Regression task
        openml.tasks.functions.get_task(5001)
        # Learning curve
        openml.tasks.functions.get_task(64)
        # Issue 538, get_task failing with clustering task.
        openml.tasks.functions.get_task(126033)

    @pytest.mark.test_server
    def test_download_split(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        split = task.download_split()
        assert type(split) == OpenMLSplit  # noqa: E721
        assert _cached_split_path(1).exists()

    @pytest.mark.test_server
    def test_download_split_replaces_corrupted_file(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        expected = task.download_split()

        split_file = _cached_split_path(1)
        split_file.with_suffix(".pkl.py3").unlink()
        split_file.write_text(
            "@relation corrupted\n"
            "@attribute type {TRAIN,TEST}\n"
            "@attribute rowid INTEGER\n"
            "@attribute repeat INTEGER\n"
            "@attribute fold INTEGER\n"
            "@data\n"
            "TRAIN,1\n"
        )

        assert task.download_split() == expected
        assert "corrupted" not in split_file.read_text()

    @pytest.mark.test_server
    def test_get_task_with_download_splits(self):
        openml.tasks.get_task(1, download_splits=True)
        assert _cached_split_path(1).exists()


@mock.patch.object(requests.Session, "request")
def test_delete_task_not_owned(mock_request, test_files_directory, test_server_v1, test_apikey_v1):
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_not_owned.xml"
    mock_request.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )
    with pytest.raises(
        OpenMLNotAuthorizedError,
        match=r"The task can not be deleted because it was not uploaded by you.",
    ):
        openml.tasks.delete_task(1)

    task_url = test_server_v1 + "task/1"
    assert task_url == mock_request.call_args.kwargs.get("url")
    assert test_apikey_v1 == mock_request.call_args.kwargs.get("params", {}).get("api_key")


@mock.patch.object(requests.Session, "request")
def test_delete_task_with_run(mock_request, test_files_directory, test_server_v1, test_apikey_v1):
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_has_runs.xml"
    mock_request.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The task can not be deleted because it still has associated entities:",
    ):
        openml.tasks.delete_task(3496)

    task_url = test_server_v1 + "task/3496"
    assert task_url == mock_request.call_args.kwargs.get("url")
    assert test_apikey_v1 == mock_request.call_args.kwargs.get("params", {}).get("api_key")


@mock.patch.object(requests.Session, "request")
def test_delete_success(mock_request, test_files_directory, test_server_v1, test_apikey_v1):
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_successful.xml"
    mock_request.return_value = create_request_response(
        status_code=200,
        content_filepath=content_file,
    )

    success = openml.tasks.delete_task(361323)
    assert success

    task_url = test_server_v1 + "task/361323"
    assert task_url == mock_request.call_args.kwargs.get("url")
    assert test_apikey_v1 == mock_request.call_args.kwargs.get("params", {}).get("api_key")


@mock.patch.object(requests.Session, "request")
def test_delete_unknown_task(mock_request, test_files_directory, test_server_v1, test_apikey_v1):
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_not_exist.xml"
    mock_request.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLServerException,
        match="Task does not exist",
    ):
        openml.tasks.delete_task(9_999_999)

    task_url = test_server_v1 + "task/9999999"
    assert task_url == mock_request.call_args.kwargs.get("url")
    assert test_apikey_v1 == mock_request.call_args.kwargs.get("params", {}).get("api_key")
