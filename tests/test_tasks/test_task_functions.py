# License: BSD 3-Clause

import os
from unittest import mock

import pytest
import requests

from openml.tasks import TaskType
from openml.testing import TestBase, create_request_response
from openml import OpenMLSplit, OpenMLTask
from openml.exceptions import OpenMLCacheException, OpenMLNotAuthorizedError, OpenMLServerException
import openml
import unittest
import pandas as pd


class TestTask(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super(TestTask, self).setUp()

    def tearDown(self):
        super(TestTask, self).tearDown()

    def test__get_cached_tasks(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        tasks = openml.tasks.functions._get_cached_tasks()
        self.assertIsInstance(tasks, dict)
        self.assertEqual(len(tasks), 3)
        self.assertIsInstance(list(tasks.values())[0], OpenMLTask)

    def test__get_cached_task(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.functions._get_cached_task(1)
        self.assertIsInstance(task, OpenMLTask)

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
        self.assertIsInstance(estimation_procedures, list)
        self.assertIsInstance(estimation_procedures[0], dict)
        self.assertEqual(
            estimation_procedures[0]["task_type_id"], TaskType.SUPERVISED_CLASSIFICATION
        )

    def test_list_clustering_task(self):
        # as shown by #383, clustering tasks can give list/dict casting problems
        openml.config.server = self.production_server
        openml.tasks.list_tasks(task_type=TaskType.CLUSTERING, size=10)
        # the expected outcome is that it doesn't crash. No assertions.

    def _check_task(self, task):
        self.assertEqual(type(task), dict)
        self.assertGreaterEqual(len(task), 2)
        self.assertIn("did", task)
        self.assertIsInstance(task["did"], int)
        self.assertIn("status", task)
        self.assertIsInstance(task["status"], str)
        self.assertIn(task["status"], ["in_preparation", "active", "deactivated"])

    def test_list_tasks_by_type(self):
        num_curves_tasks = 198  # number is flexible, check server if fails
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid)
        self.assertGreaterEqual(len(tasks), num_curves_tasks)
        for tid in tasks:
            self.assertEqual(ttid, tasks[tid]["ttid"])
            self._check_task(tasks[tid])

    def test_list_tasks_output_format(self):
        ttid = TaskType.LEARNING_CURVE
        tasks = openml.tasks.list_tasks(task_type=ttid, output_format="dataframe")
        self.assertIsInstance(tasks, pd.DataFrame)
        self.assertGreater(len(tasks), 100)

    def test_list_tasks_empty(self):
        tasks = openml.tasks.list_tasks(tag="NoOneWillEverUseThisTag")
        if len(tasks) > 0:
            raise ValueError("UnitTest Outdated, got somehow results (tag is used, please adapt)")

        self.assertIsInstance(tasks, dict)

    def test_list_tasks_by_tag(self):
        num_basic_tasks = 100  # number is flexible, check server if fails
        tasks = openml.tasks.list_tasks(tag="OpenML100")
        self.assertGreaterEqual(len(tasks), num_basic_tasks)
        for tid in tasks:
            self._check_task(tasks[tid])

    def test_list_tasks(self):
        tasks = openml.tasks.list_tasks()
        self.assertGreaterEqual(len(tasks), 900)
        for tid in tasks:
            self._check_task(tasks[tid])

    def test_list_tasks_paginate(self):
        size = 10
        max = 100
        for i in range(0, max, size):
            tasks = openml.tasks.list_tasks(offset=i, size=size)
            self.assertGreaterEqual(size, len(tasks))
            for tid in tasks:
                self._check_task(tasks[tid])

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
                self.assertGreaterEqual(size, len(tasks))
                for tid in tasks:
                    self.assertEqual(j, tasks[tid]["ttid"])
                    self._check_task(tasks[tid])

    def test__get_task(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        openml.tasks.get_task(1882)

    @unittest.skip(
        "Please await outcome of discussion: https://github.com/openml/OpenML/issues/776"
    )  # noqa: E501
    def test__get_task_live(self):
        # Test the following task as it used to throw an Unicode Error.
        # https://github.com/openml/openml-python/issues/378
        openml.config.server = self.production_server
        openml.tasks.get_task(34536)

    def test_get_task(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        self.assertIsInstance(task, OpenMLTask)
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.workdir,
                    "org",
                    "openml",
                    "test",
                    "tasks",
                    "1",
                    "task.xml",
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.workdir, "org", "openml", "test", "tasks", "1", "datasplits.arff")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.workdir, "org", "openml", "test", "datasets", "1", "dataset.arff")
            )
        )

    def test_get_task_lazy(self):
        task = openml.tasks.get_task(2, download_data=False)  # anneal; crossvalidation
        self.assertIsInstance(task, OpenMLTask)
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.workdir,
                    "org",
                    "openml",
                    "test",
                    "tasks",
                    "2",
                    "task.xml",
                )
            )
        )
        self.assertEqual(task.class_labels, ["1", "2", "3", "4", "5", "U"])

        self.assertFalse(
            os.path.exists(
                os.path.join(self.workdir, "org", "openml", "test", "tasks", "2", "datasplits.arff")
            )
        )
        # Since the download_data=False is propagated to get_dataset
        self.assertFalse(
            os.path.exists(
                os.path.join(self.workdir, "org", "openml", "test", "datasets", "2", "dataset.arff")
            )
        )

        task.download_split()
        self.assertTrue(
            os.path.exists(
                os.path.join(self.workdir, "org", "openml", "test", "tasks", "2", "datasplits.arff")
            )
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
        self.assertFalse(os.path.exists(os.path.join(os.getcwd(), "tasks", "1", "tasks.xml")))

    def test_get_task_with_cache(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1)
        self.assertIsInstance(task, OpenMLTask)

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
        self.assertEqual(type(split), OpenMLSplit)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.workdir, "org", "openml", "test", "tasks", "1", "datasplits.arff")
            )
        )

    def test_deletion_of_cache_dir(self):
        # Simple removal
        tid_cache_dir = openml.utils._create_cache_directory_for_id(
            "tasks",
            1,
        )
        self.assertTrue(os.path.exists(tid_cache_dir))
        openml.utils._remove_cache_dir_for_id("tasks", tid_cache_dir)
        self.assertFalse(os.path.exists(tid_cache_dir))


@mock.patch.object(requests.Session, "delete")
def test_delete_task_not_owned(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "tasks" / "task_delete_not_owned.xml"
    mock_delete.return_value = create_request_response(
        status_code=412, content_filepath=content_file
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
        status_code=412, content_filepath=content_file
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
        status_code=200, content_filepath=content_file
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
        status_code=412, content_filepath=content_file
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
