# License: BSD 3-Clause
from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest

import openml
from openml import OpenMLTask
from openml.exceptions import OpenMLNotAuthorizedError, OpenMLServerException
from openml.tasks import TaskType
from openml.testing import TestBase, create_request_response


class TestTask(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @pytest.mark.production()
    @pytest.mark.xfail(reason="failures_issue_1544", strict=False)
    def test_list_clustering_task(self):
        self.use_production_server()
        # as shown by #383, clustering tasks can give list/dict casting problems
        openml.tasks.list_tasks(task_type=TaskType.CLUSTERING, size=10)
        # the expected outcome is that it doesn't crash. No assertions.

    def _check_task(self, task):
        assert type(task) == dict
        assert len(task) >= 2
        assert "did" in task
        assert isinstance(task["did"], int)
        assert "status" in task
        assert isinstance(task["status"], str)
        assert task["status"] in ["in_preparation", "active", "deactivated"]

    @pytest.mark.uses_test_server()
    def test_list_tasks_by_type(self):
        num_curves_tasks = 198  # number is flexible, check server if fails
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid)
        assert len(tasks) >= num_curves_tasks
        for task in tasks.to_dict(orient="index").values():
            assert ttid == task["ttid"]
            self._check_task(task)

    @pytest.mark.uses_test_server()
    def test_list_tasks_length(self):
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid)
        assert len(tasks) > 100

    @pytest.mark.uses_test_server()
    def test_list_tasks_empty(self):
        tasks = openml.tasks.list_tasks(tag="NoOneWillEverUseThisTag")
        assert tasks.empty

    @pytest.mark.uses_test_server()
    def test_list_tasks_by_tag(self):
        num_basic_tasks = 100  # number is flexible, check server if fails
        tasks = openml.tasks.list_tasks(tag="OpenML100")
        assert len(tasks) >= num_basic_tasks
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)

    @pytest.mark.uses_test_server()
    def test_list_tasks(self):
        tasks = openml.tasks.list_tasks()
        assert len(tasks) >= 900
        for task in tasks.to_dict(orient="index").values():
            self._check_task(task)

    @pytest.mark.uses_test_server()
    def test_list_tasks_paginate(self):
        size = 10
        max = 100
        for i in range(0, max, size):
            tasks = openml.tasks.list_tasks(offset=i, size=size)
            assert size >= len(tasks)
            for task in tasks.to_dict(orient="index").values():
                self._check_task(task)

    @pytest.mark.uses_test_server()
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

    @unittest.skip(
        "Please await outcome of discussion: https://github.com/openml/OpenML/issues/776",
    )
    @pytest.mark.production()
    def test__get_task_live(self):
        self.use_production_server()
        # Test the following task as it used to throw an Unicode Error.
        # https://github.com/openml/openml-python/issues/378
        openml.tasks.get_task(34536)

    @pytest.mark.uses_test_server()
    def test_get_task(self):
        task = openml.tasks.get_task(1, download_data=True)  # anneal; crossvalidation
        assert isinstance(task, OpenMLTask)

    @pytest.mark.uses_test_server()
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

@mock.patch("openml._api.clients.http.HTTPClient.delete")
def test_delete_task_not_owned(mock_delete):
    openml.config.start_using_configuration_for_example()
    mock_delete.side_effect = OpenMLNotAuthorizedError(
        "The task can not be deleted because it was not uploaded by you."
    )
    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The task can not be deleted because it was not uploaded by you.",
    ):
        openml.tasks.delete_task(1)

    task_url = "task/1"
    assert task_url == mock_delete.call_args.args[0]

@mock.patch("openml._api.clients.http.HTTPClient.delete")
def test_delete_task_with_run(mock_delete):
    openml.config.start_using_configuration_for_example()
    mock_delete.side_effect = OpenMLServerException("Task does not exist")

    with pytest.raises(
        OpenMLServerException,
        match="Task does not exist",
    ):
        openml.tasks.delete_task(3496)

    task_url = "task/3496"
    assert task_url == mock_delete.call_args.args[0]

@mock.patch("openml._api.clients.http.HTTPClient.delete")
def test_delete_success(mock_delete, test_files_directory):
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_successful.xml"
    mock_delete.return_value = create_request_response(
        status_code=200,
        content_filepath=content_file,
    )

    success = openml.tasks.delete_task(361323)
    assert success

    task_url = "task/361323"
    assert task_url == mock_delete.call_args.args[0]

@mock.patch("openml._api.clients.http.HTTPClient.delete")
def test_delete_unknown_task(mock_delete):
    mock_delete.side_effect = OpenMLServerException("Task does not exist")
    with pytest.raises(OpenMLServerException, match="Task does not exist"):
        openml.tasks.delete_task(9_999_999)

    task_url = "task/9999999"
    assert task_url == mock_delete.call_args.args[0]