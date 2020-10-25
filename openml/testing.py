# License: BSD 3-Clause

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
from openml.tasks import TaskType

import logging


class TestBase(unittest.TestCase):
    """Base class for tests

    Note
    ----
    Currently hard-codes a read-write key.
    Hopefully soon allows using a test server, not the production server.
    """

    publish_tracker = {
        "run": [],
        "data": [],
        "flow": [],
        "task": [],
        "study": [],
        "user": [],
    }  # type: dict
    test_server = "https://test.openml.org/api/v1/xml"
    # amueller's read/write key that he will throw away later
    apikey = "610344db6388d9ba34f6db45a3cf71de"

    # creating logger for tracking files uploaded to test server
    logger = logging.getLogger("unit_tests_published_entities")
    logger.setLevel(logging.DEBUG)

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
            static_cache_dir = os.path.abspath(os.path.join(static_cache_dir, ".."))
        content = os.listdir(static_cache_dir)
        if "files" in content:
            self.static_cache_dir = os.path.join(static_cache_dir, "files")

        if self.static_cache_dir is None:
            raise ValueError(
                "Cannot find test cache dir, expected it to be {}!".format(static_cache_dir)
            )

        self.cwd = os.getcwd()
        workdir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir_name = self.id()
        self.workdir = os.path.join(workdir, tmp_dir_name)
        shutil.rmtree(self.workdir, ignore_errors=True)

        os.mkdir(self.workdir)
        os.chdir(self.workdir)

        self.cached = True
        openml.config.apikey = TestBase.apikey
        self.production_server = "https://openml.org/api/v1/xml"
        openml.config.server = TestBase.test_server
        openml.config.avoid_duplicate_runs = False
        openml.config.cache_directory = self.workdir

        # If we're on travis, we save the api key in the config file to allow
        # the notebook tests to read them.
        if os.environ.get("TRAVIS") or os.environ.get("APPVEYOR"):
            with lockutils.external_lock("config", lock_path=self.workdir):
                with open(openml.config.config_file, "w") as fh:
                    fh.write("apikey = %s" % openml.config.apikey)

        # Increase the number of retries to avoid spurious server failures
        self.connection_n_retries = openml.config.connection_n_retries
        openml.config.connection_n_retries = 10

    def tearDown(self):
        os.chdir(self.cwd)
        try:
            shutil.rmtree(self.workdir)
        except PermissionError:
            if os.name == "nt":
                # one of the files may still be used by another process
                pass
            else:
                raise
        openml.config.server = self.production_server
        openml.config.connection_n_retries = self.connection_n_retries

    @classmethod
    def _mark_entity_for_removal(self, entity_type, entity_id):
        """ Static record of entities uploaded to test server

        Dictionary of lists where the keys are 'entity_type'.
        Each such dictionary is a list of integer IDs.
        For entity_type='flow', each list element is a tuple
        of the form (Flow ID, Flow Name).
        """
        if entity_type not in TestBase.publish_tracker:
            TestBase.publish_tracker[entity_type] = [entity_id]
        else:
            TestBase.publish_tracker[entity_type].append(entity_id)

    @classmethod
    def _delete_entity_from_tracker(self, entity_type, entity):
        """ Deletes entity records from the static file_tracker

        Given an entity type and corresponding ID, deletes all entries, including
        duplicate entries of the ID for the entity type.
        """
        if entity_type in TestBase.publish_tracker:
            # removes duplicate entries
            TestBase.publish_tracker[entity_type] = list(set(TestBase.publish_tracker[entity_type]))
            if entity_type == "flow":
                delete_index = [
                    i
                    for i, (id_, _) in enumerate(TestBase.publish_tracker[entity_type])
                    if id_ == entity
                ][0]
            else:
                delete_index = [
                    i
                    for i, id_ in enumerate(TestBase.publish_tracker[entity_type])
                    if id_ == entity
                ][0]
            TestBase.publish_tracker[entity_type].pop(delete_index)

    def _get_sentinel(self, sentinel=None):
        if sentinel is None:
            # Create a unique prefix for the flow. Necessary because the flow
            # is identified by its name and external version online. Having a
            # unique name allows us to publish the same flow in each test run.
            md5 = hashlib.md5()
            md5.update(str(time.time()).encode("utf-8"))
            md5.update(str(os.getpid()).encode("utf-8"))
            sentinel = md5.hexdigest()[:10]
            sentinel = "TEST%s" % sentinel
        return sentinel

    def _add_sentinel_to_flow_name(self, flow, sentinel=None):
        sentinel = self._get_sentinel(sentinel=sentinel)
        flows_to_visit = list()
        flows_to_visit.append(flow)
        while len(flows_to_visit) > 0:
            current_flow = flows_to_visit.pop()
            current_flow.name = "%s%s" % (sentinel, current_flow.name)
            for subflow in current_flow.components.values():
                flows_to_visit.append(subflow)

        return flow, sentinel

    def _check_dataset(self, dataset):
        self.assertEqual(type(dataset), dict)
        self.assertGreaterEqual(len(dataset), 2)
        self.assertIn("did", dataset)
        self.assertIsInstance(dataset["did"], int)
        self.assertIn("status", dataset)
        self.assertIsInstance(dataset["status"], str)
        self.assertIn(dataset["status"], ["in_preparation", "active", "deactivated"])

    def _check_fold_timing_evaluations(
        self,
        fold_evaluations: Dict,
        num_repeats: int,
        num_folds: int,
        max_time_allowed: float = 60000.0,
        task_type: TaskType = TaskType.SUPERVISED_CLASSIFICATION,
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
            "usercpu_time_millis_testing": (0, max_time_allowed),
            "usercpu_time_millis_training": (0, max_time_allowed),
            "usercpu_time_millis": (0, max_time_allowed),
            "wall_clock_time_millis_training": (0, max_time_allowed),
            "wall_clock_time_millis_testing": (0, max_time_allowed),
            "wall_clock_time_millis": (0, max_time_allowed),
        }

        if check_scores:
            if task_type in (TaskType.SUPERVISED_CLASSIFICATION, TaskType.LEARNING_CURVE):
                check_measures["predictive_accuracy"] = (0, 1.0)
            elif task_type == TaskType.SUPERVISED_REGRESSION:
                check_measures["mean_absolute_error"] = (0, float("inf"))

        self.assertIsInstance(fold_evaluations, dict)
        if sys.version_info[:2] >= (3, 3):
            # this only holds if we are allowed to record time (otherwise some
            # are missing)
            self.assertEqual(set(fold_evaluations.keys()), set(check_measures.keys()))

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


try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer


class CustomImputer(SimpleImputer):
    """Duplicate class alias for sklearn's SimpleImputer

    Helps bypass the sklearn extension duplicate operation check
    """

    pass


def cont(X):
    return X.dtypes != "category"


def cat(X):
    return X.dtypes == "category"


__all__ = ["TestBase", "SimpleImputer", "CustomImputer", "cat", "cont"]
