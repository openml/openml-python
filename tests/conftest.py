"""This file is recognized by pytest for defining specified behaviour

'conftest.py' files are directory-scope files that are shared by all
sub-directories from where this file is placed. pytest recognises
'conftest.py' for any unit test executed from within this directory
tree. This file is used to define fixtures, hooks, plugins, and other
functionality that can be shared by the unit tests.

This file has been created for the OpenML testing to primarily make use
of the pytest hooks 'pytest_sessionstart' and 'pytest_sessionfinish',
which are being used for managing the deletion of local and remote files
created by the unit tests, run across more than one process.

This design allows one to comment or remove the conftest.py file to
disable file deletions, without editing any of the test case files.


Possible Future: class TestBase from openml/testing.py can be included
    under this file and there would not be any requirements to import
    testing.py in each of the unit test modules.
"""

# License: BSD 3-Clause
from __future__ import annotations

import logging
import os
from pathlib import Path
import pytest

import openml
from openml.testing import TestBase

# creating logger for unit test file deletion status
logger = logging.getLogger("unit_tests")
logger.setLevel(logging.DEBUG)

file_list = []


def worker_id() -> str:
    """Returns the name of the worker process owning this function call.

    :return: str
        Possible outputs from the set of {'master', 'gw0', 'gw1', ..., 'gw(n-1)'}
        where n is the number of workers being used by pytest-xdist
    """
    vars_ = list(os.environ.keys())
    if "PYTEST_XDIST_WORKER" in vars_ or "PYTEST_XDIST_WORKER_COUNT" in vars_:
        return os.environ["PYTEST_XDIST_WORKER"]
    else:
        return "master"


def read_file_list() -> list[Path]:
    """Returns a list of paths to all files that currently exist in 'openml/tests/files/'

    :return: List[Path]
    """
    test_files_dir = Path(__file__).parent / "files"
    return [f for f in test_files_dir.rglob("*") if f.is_file()]


def compare_delete_files(old_list: list[Path], new_list: list[Path]) -> None:
    """Deletes files that are there in the new_list but not in the old_list

    :param old_list: List[Path]
    :param new_list: List[Path]
    :return: None
    """
    file_list = list(set(new_list) - set(old_list))
    for file in file_list:
        os.remove(file)
        logger.info(f"Deleted from local: {file}")


def delete_remote_files(tracker, flow_names) -> None:
    """Function that deletes the entities passed as input, from the OpenML test server

    The TestBase class in openml/testing.py has an attribute called publish_tracker.
    This function expects the dictionary of the same structure.
    It is a dictionary of lists, where the keys are entity types, while the values are
    lists of integer IDs, except for key 'flow' where the value is a tuple (ID, flow name).

    Iteratively, multiple POST requests are made to the OpenML test server using
    openml.utils._delete_entity() to remove the entities uploaded by all the unit tests.

    :param tracker: Dict
    :return: None
    """
    openml.config.server = TestBase.test_server
    openml.config.apikey = TestBase.apikey

    # reordering to delete sub flows at the end of flows
    # sub-flows have shorter names, hence, sorting by descending order of flow name length
    if "flow" in tracker:
        to_sort = list(zip(tracker["flow"], flow_names))
        flow_deletion_order = [
            entity_id for entity_id, _ in sorted(to_sort, key=lambda x: len(x[1]), reverse=True)
        ]
        tracker["flow"] = [flow_deletion_order[1] for flow_id, _ in flow_deletion_order]

    # deleting all collected entities published to test server
    # 'run's are deleted first to prevent dependency issue of entities on deletion
    logger.info("Entity Types: {}".format(["run", "data", "flow", "task", "study"]))
    for entity_type in ["run", "data", "flow", "task", "study"]:
        logger.info(f"Deleting {entity_type}s...")
        for _i, entity in enumerate(tracker[entity_type]):
            try:
                openml.utils._delete_entity(entity_type, entity)
                logger.info(f"Deleted ({entity_type}, {entity})")
            except Exception as e:
                logger.warning(f"Cannot delete ({entity_type},{entity}): {e}")


def pytest_sessionstart() -> None:
    """Pytest hook that is executed before any unit test starts

    This function will be called by each of the worker processes, along with the master process
    when they are spawned. This happens even before the collection of unit tests.
    If number of workers, n=4, there will be a total of 5 (1 master + 4 workers) calls of this
    function, before execution of any unit test begins. The master pytest process has the name
    'master' while the worker processes are named as 'gw{i}' where i = 0, 1, ..., n-1.
    The order of process spawning is: 'master' -> random ordering of the 'gw{i}' workers.

    Since, master is always executed first, it is checked if the current process is 'master' and
    store a list of strings of paths of all files in the directory (pre-unit test snapshot).

    :return: None
    """
    # file_list is global to maintain the directory snapshot during tear down
    global file_list
    worker = worker_id()
    if worker == "master":
        file_list = read_file_list()


def pytest_sessionfinish() -> None:
    """Pytest hook that is executed after all unit tests of a worker ends

    This function will be called by each of the worker processes, along with the master process
    when they are done with the unit tests allocated to them.
    If number of workers, n=4, there will be a total of 5 (1 master + 4 workers) calls of this
    function, before execution of any unit test begins. The master pytest process has the name
    'master' while the worker processes are named as 'gw{i}' where i = 0, 1, ..., n-1.
    The order of invocation is: random ordering of the 'gw{i}' workers -> 'master'.

    Since, master is always executed last, it is checked if the current process is 'master' and,
    * Compares file list with pre-unit test snapshot and deletes all local files generated
    * Iterates over the list of entities uploaded to test server and deletes them remotely

    :return: None
    """
    # allows access to the file_list read in the set up phase
    global file_list
    worker = worker_id()
    logger.info(f"Finishing worker {worker}")

    # Test file deletion
    logger.info(f"Deleting files uploaded to test server for worker {worker}")
    delete_remote_files(TestBase.publish_tracker, TestBase.flow_name_tracker)

    if worker == "master":
        # Local file deletion
        new_file_list = read_file_list()
        compare_delete_files(file_list, new_file_list)
        logger.info("Local files deleted")

    logger.info(f"{worker} is killed")


def pytest_configure(config):
    config.addinivalue_line("markers", "sklearn: marks tests that use scikit-learn")


def pytest_addoption(parser):
    parser.addoption(
        "--long",
        action="store_true",
        default=False,
        help="Run the long version of tests which support both short and long scenarios.",
    )


def _expected_static_cache_state(root_dir: Path) -> list[Path]:
    _c_root_dir = root_dir / "org" / "openml" / "test"
    res_paths = [root_dir, _c_root_dir]
    
    for _d in ["datasets", "tasks", "runs", "setups"]:
        res_paths.append(_c_root_dir / _d)

    for _id in ["-1","2"]:
        tmp_p = _c_root_dir / "datasets" / _id
        res_paths.extend([
            tmp_p / "dataset.arff",
            tmp_p / "features.xml",
            tmp_p / "qualities.xml",
            tmp_p / "description.xml",
        ])

    res_paths.append(_c_root_dir / "datasets" / "30" / "dataset_30.pq")
    res_paths.append(_c_root_dir / "runs" / "1" / "description.xml")
    res_paths.append(_c_root_dir / "setups" / "1" / "description.xml")
    
    for _id in ["1", "3", "1882"]:
        tmp_p = _c_root_dir / "tasks" / _id
        res_paths.extend([
            tmp_p / "datasplits.arff",
            tmp_p / "task.xml",
        ])
    
    return res_paths


def assert_static_test_cache_correct(root_dir: Path) -> None:
    for p in _expected_static_cache_state(root_dir):
        assert p.exists(), f"Expected path {p} does not exist"
    

@pytest.fixture(scope="class")
def long_version(request):
    request.cls.long_version = request.config.getoption("--long")


@pytest.fixture()
def test_files_directory() -> Path:
    return Path(__file__).parent / "files"


@pytest.fixture()
def test_api_key() -> str:
    return "c0c42819af31e706efe1f4b88c23c6c1"


@pytest.fixture(autouse=True)
def verify_cache_state(test_files_directory) -> None:
    assert_static_test_cache_correct(test_files_directory)
    yield
    assert_static_test_cache_correct(test_files_directory)
