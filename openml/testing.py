# License: BSD 3-Clause
from __future__ import annotations

import hashlib
import inspect
import logging
import os
import pathlib
import shutil
import time
import unittest
from pathlib import Path
from typing import ClassVar

import pandas as pd
import requests

import openml
from openml.exceptions import OpenMLServerException
from openml.tasks import TaskType


def _check_dataset(dataset: dict) -> None:
    assert isinstance(dataset, dict)
    assert len(dataset) >= 2
    assert "did" in dataset
    assert isinstance(dataset["did"], int)
    assert "status" in dataset
    assert dataset["status"] in ["in_preparation", "active", "deactivated"]


class TestBase(unittest.TestCase):
    """Base class for tests

    Note
    ----
    Currently hard-codes a read-write key.
    Hopefully soon allows using a test server, not the production server.
    """

    # TODO: This could be made more explcit with a TypedDict instead of list[str | int]
    publish_tracker: ClassVar[dict[str, list[str | int]]] = {
        "run": [],
        "data": [],
        "flow": [],
        "task": [],
        "study": [],
        "user": [],
    }
    flow_name_tracker: ClassVar[list[str]] = []
    test_server = "https://test.openml.org/api/v1/xml"
    # amueller's read/write key that he will throw away later
    apikey = "610344db6388d9ba34f6db45a3cf71de"

    # creating logger for tracking files uploaded to test server
    logger = logging.getLogger("unit_tests_published_entities")
    logger.setLevel(logging.DEBUG)

    def setUp(self, n_levels: int = 1) -> None:
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
        abspath_this_file = Path(inspect.getfile(self.__class__)).absolute()
        static_cache_dir = abspath_this_file.parent
        for _ in range(n_levels):
            static_cache_dir = static_cache_dir.parent.absolute()

        content = os.listdir(static_cache_dir)
        if "files" in content:
            static_cache_dir = static_cache_dir / "files"
        else:
            raise ValueError(
                f"Cannot find test cache dir, expected it to be {static_cache_dir}!",
            )

        self.static_cache_dir = static_cache_dir
        self.cwd = Path.cwd()
        workdir = Path(__file__).parent.absolute()
        tmp_dir_name = self.id()
        self.workdir = workdir / tmp_dir_name
        shutil.rmtree(self.workdir, ignore_errors=True)

        self.workdir.mkdir(exist_ok=True)
        os.chdir(self.workdir)

        self.cached = True
        openml.config.apikey = TestBase.apikey
        self.production_server = "https://openml.org/api/v1/xml"
        openml.config.server = TestBase.test_server
        openml.config.avoid_duplicate_runs = False
        openml.config.set_root_cache_directory(str(self.workdir))

        # Increase the number of retries to avoid spurious server failures
        self.retry_policy = openml.config.retry_policy
        self.connection_n_retries = openml.config.connection_n_retries
        openml.config.set_retry_policy("robot", n_retries=20)

    def tearDown(self) -> None:
        """Tear down the test"""
        os.chdir(self.cwd)
        try:
            shutil.rmtree(self.workdir)
        except PermissionError as e:
            if os.name != "nt":
                # one of the files may still be used by another process
                raise e

        openml.config.server = self.production_server
        openml.config.connection_n_retries = self.connection_n_retries
        openml.config.retry_policy = self.retry_policy

    @classmethod
    def _mark_entity_for_removal(
        cls,
        entity_type: str,
        entity_id: int,
        entity_name: str | None = None,
    ) -> None:
        """Static record of entities uploaded to test server

        Dictionary of lists where the keys are 'entity_type'.
        Each such dictionary is a list of integer IDs.
        For entity_type='flow', each list element is a tuple
        of the form (Flow ID, Flow Name).
        """
        if entity_type not in TestBase.publish_tracker:
            TestBase.publish_tracker[entity_type] = [entity_id]
        else:
            TestBase.publish_tracker[entity_type].append(entity_id)
        if isinstance(entity_type, openml.flows.OpenMLFlow):
            assert entity_name is not None
            cls.flow_name_tracker.append(entity_name)

    @classmethod
    def _delete_entity_from_tracker(cls, entity_type: str, entity: int) -> None:
        """Deletes entity records from the static file_tracker

        Given an entity type and corresponding ID, deletes all entries, including
        duplicate entries of the ID for the entity type.
        """
        if entity_type in TestBase.publish_tracker:
            # removes duplicate entries
            TestBase.publish_tracker[entity_type] = list(set(TestBase.publish_tracker[entity_type]))
            if entity_type == "flow":
                delete_index = next(
                    i
                    for i, (id_, _) in enumerate(
                        zip(TestBase.publish_tracker[entity_type], TestBase.flow_name_tracker),
                    )
                    if id_ == entity
                )
            else:
                delete_index = next(
                    i
                    for i, id_ in enumerate(TestBase.publish_tracker[entity_type])
                    if id_ == entity
                )
            TestBase.publish_tracker[entity_type].pop(delete_index)

    def _get_sentinel(self, sentinel: str | None = None) -> str:
        if sentinel is None:
            # Create a unique prefix for the flow. Necessary because the flow
            # is identified by its name and external version online. Having a
            # unique name allows us to publish the same flow in each test run.
            md5 = hashlib.md5()  # noqa: S324
            md5.update(str(time.time()).encode("utf-8"))
            md5.update(str(os.getpid()).encode("utf-8"))
            sentinel = md5.hexdigest()[:10]
            sentinel = "TEST%s" % sentinel
        return sentinel

    def _add_sentinel_to_flow_name(
        self,
        flow: openml.flows.OpenMLFlow,
        sentinel: str | None = None,
    ) -> tuple[openml.flows.OpenMLFlow, str]:
        sentinel = self._get_sentinel(sentinel=sentinel)
        flows_to_visit = []
        flows_to_visit.append(flow)
        while len(flows_to_visit) > 0:
            current_flow = flows_to_visit.pop()
            current_flow.name = f"{sentinel}{current_flow.name}"
            for subflow in current_flow.components.values():
                flows_to_visit.append(subflow)

        return flow, sentinel

    def _check_dataset(self, dataset: dict[str, str | int]) -> None:
        _check_dataset(dataset)
        assert isinstance(dataset, dict)
        assert len(dataset) >= 2
        assert "did" in dataset
        assert isinstance(dataset["did"], int)
        assert "status" in dataset
        assert isinstance(dataset["status"], str)
        assert dataset["status"] in ["in_preparation", "active", "deactivated"]

    def _check_fold_timing_evaluations(  # noqa: PLR0913
        self,
        fold_evaluations: dict[str, dict[int, dict[int, float]]],
        num_repeats: int,
        num_folds: int,
        *,
        max_time_allowed: float = 60000.0,
        task_type: TaskType = TaskType.SUPERVISED_CLASSIFICATION,
        check_scores: bool = True,
    ) -> None:
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

        assert isinstance(fold_evaluations, dict)
        assert set(fold_evaluations.keys()) == set(check_measures.keys())

        for measure in check_measures:
            if measure in fold_evaluations:
                num_rep_entrees = len(fold_evaluations[measure])
                assert num_rep_entrees == num_repeats
                min_val = check_measures[measure][0]
                max_val = check_measures[measure][1]
                for rep in range(num_rep_entrees):
                    num_fold_entrees = len(fold_evaluations[measure][rep])
                    assert num_fold_entrees == num_folds
                    for fold in range(num_fold_entrees):
                        evaluation = fold_evaluations[measure][rep][fold]
                        assert isinstance(evaluation, float)
                        assert evaluation >= min_val
                        assert evaluation <= max_val


def check_task_existence(
    task_type: TaskType,
    dataset_id: int,
    target_name: str,
    **kwargs: dict[str, str | int | dict[str, str | int | openml.tasks.TaskType]],
) -> int | None:
    """Checks if any task with exists on test server that matches the meta data.

    Parameter
    ---------
    task_type : openml.tasks.TaskType
    dataset_id : int
    target_name : str

    Return
    ------
    int, None
    """
    return_val = None
    tasks = openml.tasks.list_tasks(task_type=task_type, output_format="dataframe")
    assert isinstance(tasks, pd.DataFrame)
    if len(tasks) == 0:
        return None
    tasks = tasks.loc[tasks["did"] == dataset_id]
    if len(tasks) == 0:
        return None
    tasks = tasks.loc[tasks["target_feature"] == target_name]
    if len(tasks) == 0:
        return None
    task_match = []
    for task_id in tasks["tid"].to_list():
        task_match.append(task_id)
        try:
            task = openml.tasks.get_task(task_id)
        except OpenMLServerException:
            # can fail if task_id deleted by another parallely run unit test
            task_match.pop(-1)
            return_val = None
            continue
        for k, v in kwargs.items():
            if getattr(task, k) != v:
                # even if one of the meta-data key mismatches, then task_id is not a match
                task_match.pop(-1)
                break
        # if task_id is retained in the task_match list, it passed all meta key-value matches
        if len(task_match) == 1:
            return_val = task_id
            break
    if len(task_match) == 0:
        return_val = None
    return return_val


try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer


class CustomImputer(SimpleImputer):
    """Duplicate class alias for sklearn's SimpleImputer

    Helps bypass the sklearn extension duplicate operation check
    """


def create_request_response(
    *,
    status_code: int,
    content_filepath: pathlib.Path,
) -> requests.Response:
    with content_filepath.open("r") as xml_response:
        response_body = xml_response.read()

    response = requests.Response()
    response.status_code = status_code
    response._content = response_body.encode()
    return response


__all__ = [
    "TestBase",
    "SimpleImputer",
    "CustomImputer",
    "check_task_existence",
    "create_request_response",
]
