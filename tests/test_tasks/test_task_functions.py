# License: BSD 3-Clause
from __future__ import annotations

import os
import unittest
from typing import cast
from unittest import mock

import pandas as pd
import pytest
import requests

import openml
from openml import OpenMLSplit, OpenMLTask
from openml.exceptions import OpenMLCacheException, OpenMLNotAuthorizedError, OpenMLServerException
from openml.tasks import TaskType
from openml.testing import TestBase, create_request_response


class TestTask(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test__get_cached_tasks(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        tasks = openml.tasks.functions._get_cached_tasks()
        assert isinstance(tasks, dict)
        assert len(tasks) == 3
        assert isinstance(next(iter(tasks.values())), OpenMLTask)

    def test__get_cached_task(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.functions._get_cached_task(1)
        assert isinstance(task, OpenMLTask)

    def test__get_cached_task_not_cached(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        self.assertRaisesRegex(
            OpenMLCacheException,
            "Task file for tid 2 not cached",
            openml.tasks.functions._get_cached_task,
            2,
        )

    def test__get_estimation_procedure_list(self):
        estimation_procedures = openml.tasks.functions._get_estimation_procedure_list()
        assert isinstance(estimation_procedures, list)
        assert isinstance(estimation_procedures[0], dict)
        assert estimation_procedures[0]["task_type_id"] == TaskType.SUPERVISED_CLASSIFICATION

    @pytest.mark.production()
    def test_list_clustering_task(self):
        # as shown by #383, clustering tasks can give list/dict casting problems
        openml.config.server = self.production_server
        openml.tasks.list_tasks(task_type=TaskType.CLUSTERING, size=10, output_format="dataframe")
        # the expected outcome is that it doesn't crash. No assertions.

    def _check_task(self, task):
        assert type(task) == dict
        assert len(task) >= 2
        assert "did" in task
        assert isinstance(task["did"], int)
        assert "status" in task
        assert isinstance(task["status"], str)
        assert task["status"] in ["in_preparation", "active", "deactivated"]

    def test_list_tasks_by_type(self):
        num_curves_tasks = 198  # number is flexible, check server if fails
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid, output_format="dataframe")
        assert len(tasks) >= num_curves_tasks
        for task in tasks.to_dict(orient="index").values():
            assert ttid == task["ttid"]
            self._check_task(task)

    def test_list_tasks_output_format(self):
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid, output_format="dataframe")
        assert isinstance(tasks, pd.DataFrame)
        assert len(tasks) > 100

    def test_list_tasks_empty(self):
        tasks = cast(
            pd.DataFrame,
            openml.tasks.list_tasks(tag="NoOneWillEverUseThisTag", output_format="dataframe"),
        )
        assert tasks.empty

    def test_list_tasks_by_tag(self):
        num_basic_tasks = 100  # number is flexible, check server if fails
        tasks = openml.tasks.list_tasks(tag="OpenML100", output_format="dataframe")
        assert len(tasks) >= num_basic_tasks
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)

    def test_list_tasks(self):
        tasks = openml.tasks.list_tasks(output_format="dataframe")
        assert len(tasks) >= 900
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)

    def test_list_tasks_paginate(self):
        size = 10
        max = 100
        for i in range(0, max, size):
            tasks = openml.tasks.list_tasks(offset=i, size=size, output_format="dataframe")
            assert size >= len(tasks)
            for task in tasks.to_dict(orient="index").values():
                self._check_task(task)

    def test_list_tasks_per_type_paginate(self):
        size = 40
        max = 100
        task_types = [
            TaskType.SUPERVISED_CLASSIFICATION,
            TaskType.SUPERVISED_REGRESSION,
            TaskType.LEARNING_CURVE,
        ]
        for j in task_types:
            for i in range(0, max, size):
                tasks = openml.tasks.list_tasks(
                    task_type=j,
                    offset=i,
                    size=size,
                    output_format="dataframe",
                )
                assert size >= len(tasks)
                for task in tasks.to_dict(orient="index").values():
                    assert j == task["ttid"]
                    self._check_task(task)

    def test__get_task(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        openml.tasks.get_task(1882)

    @unittest.skip(
        "Please await outcome of discussion: https://github.com/openml/OpenML/issues/776",
    )
    @pytest.mark.production()
    def test__get_task_live(self):
        # Test the following task as it used to throw an Unicode Error.
        # https://github.com/openml/openml-python/issues/378
        openml.config.server = self.production_server
        openml.tasks.get_task(34536)

    def test_get_task(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        assert isinstance(task, OpenMLTask)
        assert os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "tasks", "1", "task.xml")
        )
        assert os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "tasks", "1", "datasplits.arff")
        )
        assert os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "datasets", "1", "dataset.arff")
        )

    def test_get_task_lazy(self):
        task = openml.tasks.get_task(2, download_data=False)  # anneal; crossvalidation
        assert isinstance(task, OpenMLTask)
        assert os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "tasks", "2", "task.xml")
        )
        assert task.class_labels == ["1", "2", "3", "4", "5", "U"]

        assert not os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "tasks", "2", "datasplits.arff")
        )
        # Since the download_data=False is propagated to get_dataset
        assert not os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "datasets", "2", "dataset.arff")
        )

        task.download_split()
        assert os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "tasks", "2", "datasplits.arff")
        )

    @mock.patch("openml.tasks.functions.get_dataset")
    def test_removal_upon_download_failure(self, get_dataset):
        class WeirdException(Exception):
            pass

        def assert_and_raise(*args, **kwargs):
            # Make sure that the file was created!
            assert os.path.join(os.getcwd(), "tasks", "1", "tasks.xml")
            raise WeirdException()

        get_dataset.side_effect = assert_and_raise
        try:
            openml.tasks.get_task(1)  # anneal; crossvalidation
        except WeirdException:
            pass
        # Now the file should no longer exist
        assert not os.path.exists(os.path.join(os.getcwd(), "tasks", "1", "tasks.xml"))

    def test_get_task_with_cache(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1)
        assert isinstance(task, OpenMLTask)

    @pytest.mark.production()
    def test_get_task_different_types(self):
        openml.config.server = self.production_server
        # Regression task
        openml.tasks.functions.get_task(5001)
        # Learning curve
        openml.tasks.functions.get_task(64)
        # Issue 538, get_task failing with clustering task.
        openml.tasks.functions.get_task(126033)

    def test_download_split(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        split = task.download_split()
        assert type(split) == OpenMLSplit
        assert os.path.exists(
            os.path.join(self.workdir, "org", "openml", "test", "tasks", "1", "datasplits.arff")
        )

    def test_deletion_of_cache_dir(self):
        # Simple removal
        tid_cache_dir = openml.utils._create_cache_directory_for_id(
            "tasks",
            1,
        )
        assert os.path.exists(tid_cache_dir)
        openml.utils._remove_cache_dir_for_id("tasks", tid_cache_dir)
        assert not os.path.exists(tid_cache_dir)


@mock.patch.object(requests.Session, "delete")
def test_delete_task_not_owned(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_not_owned.xml"
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The task can not be deleted because it was not uploaded by you.",
    ):
        openml.tasks.delete_task(1)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/task/1",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_task_with_run(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_has_runs.xml"
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The task can not be deleted because it still has associated entities:",
    ):
        openml.tasks.delete_task(3496)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/task/3496",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_success(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_successful.xml"
    mock_delete.return_value = create_request_response(
        status_code=200,
        content_filepath=content_file,
    )

    success = openml.tasks.delete_task(361323)
    assert success

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/task/361323",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_unknown_task(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_not_exist.xml"
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLServerException,
        match="Task does not exist",
    ):
        openml.tasks.delete_task(9_999_999)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/task/9999999",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)
