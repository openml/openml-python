# License: BSD 3-Clause

import os
from unittest import mock

from openml.testing import TestBase
from openml import OpenMLSplit, OpenMLTask
from openml.exceptions import OpenMLCacheException
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
        openml.config.cache_directory = self.static_cache_dir
        tasks = openml.tasks.functions._get_cached_tasks()
        self.assertIsInstance(tasks, dict)
        self.assertEqual(len(tasks), 3)
        self.assertIsInstance(list(tasks.values())[0], OpenMLTask)

    def test__get_cached_task(self):
        openml.config.cache_directory = self.static_cache_dir
        task = openml.tasks.functions._get_cached_task(1)
        self.assertIsInstance(task, OpenMLTask)

    def test__get_cached_task_not_cached(self):
        openml.config.cache_directory = self.static_cache_dir
        self.assertRaisesRegex(
            OpenMLCacheException,
            'Task file for tid 2 not cached',
            openml.tasks.functions._get_cached_task,
            2,
        )

    def test__get_estimation_procedure_list(self):
        estimation_procedures = openml.tasks.functions.\
            _get_estimation_procedure_list()
        self.assertIsInstance(estimation_procedures, list)
        self.assertIsInstance(estimation_procedures[0], dict)
        self.assertEqual(estimation_procedures[0]['task_type_id'], 1)

    def test_list_clustering_task(self):
        # as shown by #383, clustering tasks can give list/dict casting problems
        openml.config.server = self.production_server
        openml.tasks.list_tasks(task_type_id=5, size=10)
        # the expected outcome is that it doesn't crash. No assertions.

    def _check_task(self, task):
        self.assertEqual(type(task), dict)
        self.assertGreaterEqual(len(task), 2)
        self.assertIn('did', task)
        self.assertIsInstance(task['did'], int)
        self.assertIn('status', task)
        self.assertIsInstance(task['status'], str)
        self.assertIn(task['status'],
                      ['in_preparation', 'active', 'deactivated'])

    def test_list_tasks_by_type(self):
        num_curves_tasks = 200  # number is flexible, check server if fails
        ttid = 3
        tasks = openml.tasks.list_tasks(task_type_id=ttid)
        self.assertGreaterEqual(len(tasks), num_curves_tasks)
        for tid in tasks:
            self.assertEqual(ttid, tasks[tid]["ttid"])
            self._check_task(tasks[tid])

    def test_list_tasks_output_format(self):
        ttid = 3
        tasks = openml.tasks.list_tasks(task_type_id=ttid, output_format='dataframe')
        self.assertIsInstance(tasks, pd.DataFrame)
        self.assertGreater(len(tasks), 100)

    def test_list_tasks_empty(self):
        tasks = openml.tasks.list_tasks(tag='NoOneWillEverUseThisTag')
        if len(tasks) > 0:
            raise ValueError('UnitTest Outdated, got somehow results (tag is used, please adapt)')

        self.assertIsInstance(tasks, dict)

    def test_list_tasks_by_tag(self):
        num_basic_tasks = 100  # number is flexible, check server if fails
        tasks = openml.tasks.list_tasks(tag='OpenML100')
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
        size = 10
        max = 100
        task_types = 4
        for j in range(1, task_types):
            for i in range(0, max, size):
                tasks = openml.tasks.list_tasks(task_type_id=j, offset=i, size=size)
                self.assertGreaterEqual(size, len(tasks))
                for tid in tasks:
                    self.assertEqual(j, tasks[tid]["ttid"])
                    self._check_task(tasks[tid])

    def test__get_task(self):
        openml.config.cache_directory = self.static_cache_dir
        openml.tasks.get_task(1882)

    @unittest.skip("Please await outcome of discussion: https://github.com/openml/OpenML/issues/776")  # noqa: E501
    def test__get_task_live(self):
        # Test the following task as it used to throw an Unicode Error.
        # https://github.com/openml/openml-python/issues/378
        openml.config.server = self.production_server
        openml.tasks.get_task(34536)

    def test_get_task(self):
        task = openml.tasks.get_task(1)
        self.assertIsInstance(task, OpenMLTask)
        self.assertTrue(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "tasks", "1", "task.xml",
        )))
        self.assertTrue(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "tasks", "1", "datasplits.arff"
        )))
        self.assertTrue(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "datasets", "1", "dataset.arff"
        )))

    def test_get_task_lazy(self):
        task = openml.tasks.get_task(2, download_data=False)
        self.assertIsInstance(task, OpenMLTask)
        self.assertTrue(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "tasks", "2", "task.xml",
        )))
        self.assertEqual(task.class_labels, ['1', '2', '3', '4', '5', 'U'])

        self.assertFalse(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "tasks", "2", "datasplits.arff"
        )))
        # Since the download_data=False is propagated to get_dataset
        self.assertFalse(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "datasets", "2", "dataset.arff"
        )))

        task.download_split()
        self.assertTrue(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "tasks", "2", "datasplits.arff"
        )))

    @mock.patch('openml.tasks.functions.get_dataset')
    def test_removal_upon_download_failure(self, get_dataset):
        class WeirdException(Exception):
            pass

        def assert_and_raise(*args, **kwargs):
            # Make sure that the file was created!
            assert os.path.join(os.getcwd(), "tasks", "1", "tasks.xml")
            raise WeirdException()

        get_dataset.side_effect = assert_and_raise
        try:
            openml.tasks.get_task(1)
        except WeirdException:
            pass
        # Now the file should no longer exist
        self.assertFalse(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "tasks.xml")
        ))

    def test_get_task_with_cache(self):
        openml.config.cache_directory = self.static_cache_dir
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
        task = openml.tasks.get_task(1)
        split = task.download_split()
        self.assertEqual(type(split), OpenMLSplit)
        self.assertTrue(os.path.exists(os.path.join(
            self.workdir, 'org', 'openml', 'test', "tasks", "1", "datasplits.arff"
        )))

    def test_deletion_of_cache_dir(self):
        # Simple removal
        tid_cache_dir = openml.utils._create_cache_directory_for_id(
            'tasks', 1,
        )
        self.assertTrue(os.path.exists(tid_cache_dir))
        openml.utils._remove_cache_dir_for_id('tasks', tid_cache_dir)
        self.assertFalse(os.path.exists(tid_cache_dir))
