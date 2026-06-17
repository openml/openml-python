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

    @pytest.mark.cache()
    def test__get_cached_tasks(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        tasks = openml.tasks.functions._get_cached_tasks()
        assert isinstance(tasks, dict)
        assert len(tasks) == 3
        assert isinstance(next(iter(tasks.values())), OpenMLTask)

    @pytest.mark.cache()
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

    @mock.patch.object(requests.Session, "get")
    def test__get_estimation_procedure_list(self, mock_get):
      mock_get.return_value = create_request_response(
          status_code=200,
          content_filepath=self.static_cache_dir / "mock_responses" / "tasks" / "estimation_procedure_list.xml",
      )
      estimation_procedures = openml.tasks.functions._get_estimation_procedure_list()
      assert isinstance(estimation_procedures, list)
      assert isinstance(estimation_procedures[0], dict)
      assert estimation_procedures[0]["task_type_id"] == TaskType.SUPERVISED_CLASSIFICATION

    @mock.patch.object(requests.Session, "get")
    def test_list_clustering_task(self, mock_get):
        def side_effect(url, **kwargs):
            if "estimationprocedure" in url:
                return create_request_response(
                    status_code=200,
                    content_filepath=self.static_cache_dir / "mock_responses" / "tasks" / "estimation_procedure_list.xml",
              )
            return create_request_response(
              status_code=200,
              content_filepath=self.static_cache_dir / "mock_responses" / "tasks" / "task_list_clustering.xml",
          )
        mock_get.side_effect = side_effect
        openml.tasks.list_tasks(task_type=TaskType.CLUSTERING, size=10)

    def _check_task(self, task):
        assert type(task) == dict
        assert len(task) >= 2
        assert "did" in task
        assert isinstance(task["did"], int)
        assert "status" in task
        assert isinstance(task["status"], str)
        assert task["status"] in ["in_preparation", "active", "deactivated"]

    @mock.patch.object(requests.Session, "get")
    def test_list_tasks_by_type(self, mock_get):
        def side_effect(url, **kwargs):
            if "estimationprocedure" in url:
                return create_request_response(
                    status_code=200,
                    content_filepath=self.static_cache_dir / "mock_responses" / "tasks" / "estimation_procedure_list.xml",
                )
            return create_request_response(
                status_code=200,
                content_filepath=self.static_cache_dir / "mock_responses" / "tasks" / "task_list_type_learning_curve.xml",
            )
        mock_get.side_effect = side_effect
        num_curves_tasks =198 # number is flexible, check server if fails, must be <209
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid)
        assert len(tasks) >= num_curves_tasks
        for task in tasks.to_dict(orient="index").values():
            assert ttid == task["ttid"]
            self._check_task(task)
    
    @mock.patch.object(requests.Session, "get")
    def test_list_tasks_length(self, mock_get):
        def side_effect(url, **kwargs):
            if "estimationprocedure" in url:
                return create_request_response(
                    status_code=200,
                    content_filepath=self.static_cache_dir / "mock_responses" / "tasks" / "estimation_procedure_list.xml",
                )
            return create_request_response(
                status_code=200,
                content_filepath=self.static_cache_dir / "mock_responses" / "tasks" / "task_list_type_learning_curve.xml",
            )
        mock_get.side_effect = side_effect
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid)
        assert len(tasks) > 100

#Intentionally left unmocked because mocking would require changing request handling in functions.py; Not sure if thats what the maintainers intend.(*1)
    @pytest.mark.test_server()
    def test_list_tasks_empty(self):
        tasks = openml.tasks.list_tasks(tag="NoOneWillEverUseThisTag")
        assert tasks.empty
#Intentionally left unmocked, not sure if mocking required here.(*2)
    @pytest.mark.test_server()
    def test_list_tasks_by_tag(self):
        # Server starts with 99 active tasks with the tag, and one 'in_preparation',
        # so depending on the processing of the last dataset, there may be 99 or 100 matches.
        num_basic_tasks = 99
        tasks = openml.tasks.list_tasks(tag="OpenML100")
        assert len(tasks) >= num_basic_tasks
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)

#Intentionally left unmocked as it would require a very larege fixture(integration test maybe?)(*3)
    @pytest.mark.test_server()
    def test_list_tasks(self):
        tasks = openml.tasks.list_tasks()
        assert len(tasks) >= 900
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)
#Intentionally left unmocked as (integration test maybe?)(*4)
    @pytest.mark.test_server()
    def test_list_tasks_paginate(self):
        size = 10
        max = 100
        for i in range(0, max, size):
            tasks = openml.tasks.list_tasks(offset=i, size=size)
            assert size >= len(tasks)
            for task in tasks.to_dict(orient="index").values():
                self._check_task(task)

#Intentionally left unmocked as many html requests are made (integration test maybe?)(*5)
    @pytest.mark.test_server()
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
                tasks = openml.tasks.list_tasks(task_type=j, offset=i, size=size)
                assert size >= len(tasks)
                for task in tasks.to_dict(orient="index").values():
                    assert j == task["ttid"]
                    self._check_task(task)

#Intentionally left unmocked basis comments below(*6)       
    @pytest.mark.test_server()
    def test__get_task(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        openml.tasks.get_task(1882)

    @unittest.skip(
        "Please await outcome of discussion: https://github.com/openml/OpenML/issues/776",
    )
    @pytest.mark.production_server()
    def test__get_task_live(self):
        self.use_production_server()
        # Test the following task as it used to throw an Unicode Error.
        # https://github.com/openml/openml-python/issues/378
        openml.tasks.get_task(34536)
#Intentionally left unmocked because of io(*7)
    @pytest.mark.test_server()
    def test_get_task(self):
        task = openml.tasks.get_task(1, download_data=True)  # anneal; crossvalidation
        assert isinstance(task, OpenMLTask)
        assert os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "tasks", "1", "task.xml")
        )
        assert not os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "tasks", "1", "datasplits.arff")
        )
        assert os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "datasets", "1", "dataset_1.pq")
        )
#Intentionally left unmocked because of io(*8)
    @pytest.mark.test_server()
    def test_get_task_lazy(self):
        task = openml.tasks.get_task(2, download_data=False)  # anneal; crossvalidation
        assert isinstance(task, OpenMLTask)
        assert os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "tasks", "2", "task.xml")
        )
        assert task.class_labels == ["1", "2", "3", "4", "5", "U"]

        assert not os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "tasks", "2", "datasplits.arff")
        )
        # Since the download_data=False is propagated to get_dataset
        assert not os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "datasets", "2", "dataset.arff")
        )

        task.download_split()
        assert os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "tasks", "2", "datasplits.arff")
        )

#Intentionally left unmocked because of io(*9)
    @mock.patch("openml.tasks.functions.get_dataset")
    @pytest.mark.test_server()
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

    @pytest.mark.cache()
    def test_get_task_with_cache(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1)
        assert isinstance(task, OpenMLTask)
#intentionally left unmocked (*10)
    @pytest.mark.production_server()
    def test_get_task_different_types(self):
        self.use_production_server()
        # Regression task
        openml.tasks.functions.get_task(5001)
        # Learning curve
        openml.tasks.functions.get_task(64)
        # Issue 538, get_task failing with clustering task.
        openml.tasks.functions.get_task(126033)

#intentionally left unmocked (*11)
    @pytest.mark.test_server()
    def test_download_split(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        split = task.download_split()
        assert type(split) == OpenMLSplit
        assert os.path.exists(
            os.path.join(openml.config.get_cache_directory(), "tasks", "1", "datasplits.arff")
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
def test_delete_task_not_owned(mock_delete, test_files_directory, test_server_v1, test_apikey_v1):
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

    task_url = test_server_v1 + "task/1"
    assert task_url == mock_delete.call_args.args[0]
    assert test_apikey_v1 == mock_delete.call_args.kwargs.get("params", {}).get("api_key")


@mock.patch.object(requests.Session, "delete")
def test_delete_task_with_run(mock_delete, test_files_directory, test_server_v1, test_apikey_v1):
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

    task_url = test_server_v1 + "task/3496"
    assert task_url == mock_delete.call_args.args[0]
    assert test_apikey_v1 == mock_delete.call_args.kwargs.get("params", {}).get("api_key")


@mock.patch.object(requests.Session, "delete")
def test_delete_success(mock_delete, test_files_directory, test_server_v1, test_apikey_v1):
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_successful.xml"
    mock_delete.return_value = create_request_response(
        status_code=200,
        content_filepath=content_file,
    )

    success = openml.tasks.delete_task(361323)
    assert success

    task_url = test_server_v1 + "task/361323"
    assert task_url == mock_delete.call_args.args[0]
    assert test_apikey_v1 == mock_delete.call_args.kwargs.get("params", {}).get("api_key")


@mock.patch.object(requests.Session, "delete")
def test_delete_unknown_task(mock_delete, test_files_directory, test_server_v1, test_apikey_v1):
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

    task_url = test_server_v1 + "task/9999999"
    assert task_url == mock_delete.call_args.args[0]
    assert test_apikey_v1 == mock_delete.call_args.kwargs.get("params", {}).get("api_key")
