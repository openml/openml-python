import hashlib
import inspect
import os
import shutil
import sys
import time
from typing import Dict
import unittest
import warnings

# Currently, importing oslo raises a lot of warning that it will stop working
# under python3.8; remove this once they disappear
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from oslo_concurrency import lockutils

import openml
from openml.tasks import TaskTypeEnum

import pytest


class TestBase(unittest.TestCase):
    """Base class for tests

    Note
    ----
    Currently hard-codes a read-write key.
    Hopefully soon allows using a test server, not the production server.
    """
    tracker: Dict[str, int] = {}
    test_server = None
    apikey = None

    def setUp(self, n_levels: int = 1):
        """Setup variables and temporary directories.

        In particular, this methods:

        * creates a temporary working directory
        * figures out a path to a few static test files
        * set the default server to be the test server
        * set a static API key for the test server
        * increases the maximal number of retries

        Parameters
        ----------
        n_levels : int
            Number of nested directories the test is in. Necessary to resolve the path to the
            ``files`` directory, which is located directly under the ``tests`` directory.
        """

        # This cache directory is checked in to git to simulate a populated
        # cache
        self.maxDiff = None
        self.static_cache_dir = None
        abspath_this_file = os.path.abspath(inspect.getfile(self.__class__))
        static_cache_dir = os.path.dirname(abspath_this_file)
        for _ in range(n_levels):
            static_cache_dir = os.path.abspath(os.path.join(static_cache_dir, '..'))
        content = os.listdir(static_cache_dir)
        if 'files' in content:
            self.static_cache_dir = os.path.join(static_cache_dir, 'files')

        if self.static_cache_dir is None:
            raise ValueError('Cannot find test cache dir!')

        self.cwd = os.getcwd()
        workdir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir_name = self.id()
        self.workdir = os.path.join(workdir, tmp_dir_name)
        shutil.rmtree(self.workdir, ignore_errors=True)

        os.mkdir(self.workdir)
        os.chdir(self.workdir)

        self.cached = True
        # amueller's read/write key that he will throw away later
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        self.production_server = "https://openml.org/api/v1/xml"
        self.test_server = "https://test.openml.org/api/v1/xml"

        # For global file deletion on test server
        TestBase.test_server = self.test_server
        TestBase.apikey = openml.config.apikey

        openml.config.server = self.test_server
        openml.config.avoid_duplicate_runs = False
        openml.config.cache_directory = self.workdir

        # If we're on travis, we save the api key in the config file to allow
        # the notebook tests to read them.
        if os.environ.get('TRAVIS') or os.environ.get('APPVEYOR'):
            with lockutils.external_lock('config', lock_path=self.workdir):
                with open(openml.config.config_file, 'w') as fh:
                    fh.write('apikey = %s' % openml.config.apikey)

        # Increase the number of retries to avoid spurious server failures
        self.connection_n_retries = openml.config.connection_n_retries
        openml.config.connection_n_retries = 10

    def tearDown(self):
        os.chdir(self.cwd)
        try:
            shutil.rmtree(self.workdir)
        except PermissionError:
            if os.name == 'nt':
                # one of the files may still be used by another process
                pass
            else:
                raise
        openml.config.server = self.production_server
        openml.config.connection_n_retries = self.connection_n_retries

    @classmethod
    def _track_test_server_dumps(self, entity_type, entity_id):
        """ Static record of entities uploaded to test server

        Dictionary of lists where the keys are 'entity_type'.
        Each such dictionary is a list of integer IDs.
        For entity_type='flow', each list element is a tuple
        of the form (Flow ID, Flow Name).
        """
        if entity_type not in TestBase.tracker:
            TestBase.tracker[entity_type] = [entity_id]
        else:
            TestBase.tracker[entity_type].append(entity_id)

    @classmethod
    def _delete_entity_from_tracker(self, entity_type, entity):
        if entity_type in TestBase.tracker:
            # delete_index handles duplicate entries
            delete_index = []
            for i, element in enumerate(TestBase.tracker[entity_type]):
                if entity_type == 'flow':
                    id, name = element
                else:
                    id = element
                if id == entity:
                    delete_index.append(i)
            TestBase.tracker[entity_type] = [TestBase.tracker[entity_type][index]
                                             for index in range(len(TestBase.tracker[entity_type]))
                                             if index not in delete_index]

    @pytest.fixture(scope="session", autouse=True)
    def _cleanup_fixture(self):
        """Cleans up files generated by unit tests

        This function is called at the beginning of the invocation of
        TestBase (defined below), by each of class that inherits TestBase.
        The 'yield' creates a checkpoint and breaks away to continue running
        the unit tests of the sub class. When all the tests end, execution
        resumes from the checkpoint.
        """

        abspath_this_file = os.path.abspath(inspect.getfile(self.__class__))
        static_cache_dir = os.path.dirname(abspath_this_file)
        # Could be a risky while condition, however, going up a directory
        # n-times will eventually end at main directory
        while True:
            if 'openml' in os.listdir(static_cache_dir):
                break
            else:
                static_cache_dir = os.path.join(static_cache_dir, '../')
        directory = os.path.join(static_cache_dir, 'tests/files/')
        files = os.walk(directory)
        old_file_list = []
        for root, _, filenames in files:
            for filename in filenames:
                old_file_list.append(os.path.join(root, filename))
        # context switches to other remaining tests
        # pauses the code execution here till all tests in the 'session' is over
        yield
        # resumes from here after all collected tests are completed

        # Local file deletion
        files = os.walk(directory)
        new_file_list = []
        for root, _, filenames in files:
            for filename in filenames:
                new_file_list.append(os.path.join(root, filename))
        # filtering the files generated during this run
        new_file_list = list(set(new_file_list) - set(old_file_list))
        for file in new_file_list:
            os.remove(file)

        # Test server deletion
        openml.config.server = TestBase.test_server
        openml.config.apikey = TestBase.apikey
        entity_types = list(TestBase.tracker.keys())
        # deleting 'run' first to allow other dependent entities to be deleted
        if 'run' in entity_types:
            index = entity_types.index('run')
            # putting 'run' in the start of the list
            entity_types[0], entity_types[index] = entity_types[index], entity_types[0]

        # cloning file tracker to allow deletion of entries of deleted files
        tracker = TestBase.tracker.copy()
        # reordering to delete sub flows later
        if 'flow' in entity_types:
            flows = {}
            for entity_id, entity_name in tracker['flow']:
                flows[entity_name] = entity_id
            # reordering flows in descending order of their flow name lengths
            flow_deletion_order = [flows[name] for name in sorted(list(flows.keys()),
                                                                  key=lambda x: len(x),
                                                                  reverse=True)]
            tracker['flow'] = flow_deletion_order

        # deleting all collected entities published to test server
        for entity_type in entity_types:
            for i, entity in enumerate(tracker[entity_type]):
                try:
                    openml.utils._delete_entity(entity_type, entity)
                    print("Deleted ({}, {})".format(entity_type, entity))
                    # deleting actual entry from tracker
                    TestBase._delete_entity_from_tracker(entity_type, entity)
                except Exception as e:
                    print("Cannot delete ({}, {}): {}".format(entity_type, entity, e))
        print("End of cleanup_fixture from {}\n".format(self.__class__))

    def _get_sentinel(self, sentinel=None):
        if sentinel is None:
            # Create a unique prefix for the flow. Necessary because the flow
            # is identified by its name and external version online. Having a
            # unique name allows us to publish the same flow in each test run.
            md5 = hashlib.md5()
            md5.update(str(time.time()).encode('utf-8'))
            md5.update(str(os.getpid()).encode('utf-8'))
            sentinel = md5.hexdigest()[:10]
            sentinel = 'TEST%s' % sentinel
        return sentinel

    def _add_sentinel_to_flow_name(self, flow, sentinel=None):
        sentinel = self._get_sentinel(sentinel=sentinel)
        flows_to_visit = list()
        flows_to_visit.append(flow)
        while len(flows_to_visit) > 0:
            current_flow = flows_to_visit.pop()
            current_flow.name = '%s%s' % (sentinel, current_flow.name)
            for subflow in current_flow.components.values():
                flows_to_visit.append(subflow)

        return flow, sentinel

    def _check_dataset(self, dataset):
        self.assertEqual(type(dataset), dict)
        self.assertGreaterEqual(len(dataset), 2)
        self.assertIn('did', dataset)
        self.assertIsInstance(dataset['did'], int)
        self.assertIn('status', dataset)
        self.assertIsInstance(dataset['status'], str)
        self.assertIn(dataset['status'], ['in_preparation', 'active',
                                          'deactivated'])

    def _check_fold_timing_evaluations(
        self,
        fold_evaluations: Dict,
        num_repeats: int,
        num_folds: int,
        max_time_allowed: float = 60000.0,
        task_type: int = TaskTypeEnum.SUPERVISED_CLASSIFICATION,
        check_scores: bool = True,
    ):
        """
        Checks whether the right timing measures are attached to the run
        (before upload). Test is only performed for versions >= Python3.3

        In case of check_n_jobs(clf) == false, please do not perform this
        check (check this condition outside of this function. )
        default max_time_allowed (per fold, in milli seconds) = 1 minute,
        quite pessimistic
        """

        # a dict mapping from openml measure to a tuple with the minimum and
        # maximum allowed value
        check_measures = {
            # should take at least one millisecond (?)
            'usercpu_time_millis_testing': (0, max_time_allowed),
            'usercpu_time_millis_training': (0, max_time_allowed),
            'usercpu_time_millis': (0, max_time_allowed),
            'wall_clock_time_millis_training': (0, max_time_allowed),
            'wall_clock_time_millis_testing': (0, max_time_allowed),
            'wall_clock_time_millis': (0, max_time_allowed),
        }

        if check_scores:
            if task_type in (TaskTypeEnum.SUPERVISED_CLASSIFICATION, TaskTypeEnum.LEARNING_CURVE):
                check_measures['predictive_accuracy'] = (0, 1.)
            elif task_type == TaskTypeEnum.SUPERVISED_REGRESSION:
                check_measures['mean_absolute_error'] = (0, float("inf"))

        self.assertIsInstance(fold_evaluations, dict)
        if sys.version_info[:2] >= (3, 3):
            # this only holds if we are allowed to record time (otherwise some
            # are missing)
            self.assertEqual(set(fold_evaluations.keys()),
                             set(check_measures.keys()))

        for measure in check_measures.keys():
            if measure in fold_evaluations:
                num_rep_entrees = len(fold_evaluations[measure])
                self.assertEqual(num_rep_entrees, num_repeats)
                min_val = check_measures[measure][0]
                max_val = check_measures[measure][1]
                for rep in range(num_rep_entrees):
                    num_fold_entrees = len(fold_evaluations[measure][rep])
                    self.assertEqual(num_fold_entrees, num_folds)
                    for fold in range(num_fold_entrees):
                        evaluation = fold_evaluations[measure][rep][fold]
                        self.assertIsInstance(evaluation, float)
                        self.assertGreaterEqual(evaluation, min_val)
                        self.assertLessEqual(evaluation, max_val)


__all__ = ['TestBase']
