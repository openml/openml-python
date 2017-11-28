import os
import sys

import six

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

from openml.testing import TestBase
from openml import OpenMLSplit, OpenMLTask
from openml.exceptions import OpenMLCacheException
import openml


class TestTask(TestBase):
    _multiprocess_can_split_ = True

    def test__get_cached_tasks(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        tasks = openml.tasks.functions._get_cached_tasks()
        self.assertIsInstance(tasks, dict)
        self.assertEqual(len(tasks), 3)
        self.assertIsInstance(list(tasks.values())[0], OpenMLTask)

    def test__get_cached_task(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        task = openml.tasks.functions._get_cached_task(1)
        self.assertIsInstance(task, OpenMLTask)

    def test__get_cached_task_not_cached(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        self.assertRaisesRegexp(OpenMLCacheException,
                                'Task file for tid 2 not cached',
                                openml.tasks.functions._get_cached_task, 2)

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
        self.assertIsInstance(task['status'], six.string_types)
        self.assertIn(task['status'],
                      ['in_preparation', 'active', 'deactivated'])

    def test_list_tasks_by_type(self):
        num_curves_tasks = 200 # number is flexible, check server if fails
        ttid=3
        tasks = openml.tasks.list_tasks(task_type_id=ttid)
        self.assertGreaterEqual(len(tasks), num_curves_tasks)
        for tid in tasks:
            self.assertEquals(ttid, tasks[tid]["ttid"])
            self._check_task(tasks[tid])

    def test_list_tasks_by_tag(self):
        num_basic_tasks = 100 # number is flexible, check server if fails
        tasks = openml.tasks.list_tasks(tag='study_14')
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
        for j in range(1,task_types):
            for i in range(0, max, size):
                tasks = openml.tasks.list_tasks(task_type_id=j, offset=i, size=size)
                self.assertGreaterEqual(size, len(tasks))
                for tid in tasks:
                    self.assertEquals(j, tasks[tid]["ttid"])
                    self._check_task(tasks[tid])

    def test__get_task(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1882)

    def test_get_task(self):
        task = openml.tasks.get_task(1)
        self.assertIsInstance(task, OpenMLTask)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "task.xml")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "datasets", "1", "dataset.arff")))

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
        openml.config.set_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1)
        self.assertIsInstance(task, OpenMLTask)

    def test_download_split(self):
        task = openml.tasks.get_task(1)
        split = task.download_split()
        self.assertEqual(type(split), OpenMLSplit)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))

    def test_deletion_of_cache_dir(self):
        # Simple removal
        tid_cache_dir = openml.tasks.functions.\
            _create_task_cache_directory(1)
        self.assertTrue(os.path.exists(tid_cache_dir))
        openml.tasks.functions._remove_task_cache_dir(tid_cache_dir)
        self.assertFalse(os.path.exists(tid_cache_dir))
