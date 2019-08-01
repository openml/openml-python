'''This file is recognized by pytest for defining specified behaviour

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
'''

import os
import pickle
import logging
import posix_ipc  # required for semaphore synchronization
from typing import List

import openml
from openml.testing import TestBase

# creating logger for unit test file deletion status
logger = logging.getLogger("unit_tests")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('UnitTestDeletion.log')
logger.addHandler(fh)

file_list = []
directory = None
name = '/test'  # semaphore name
pkl_file = 'publish_tracker.pkl'  # file tracking uploaded entities


def worker_id() -> str:
    ''' Returns the name of the worker process owning this function call.

    :return: str
        Possible outputs from the set of {'master', 'gw0', 'gw1', ..., 'gw(n-1)'}
        where n is the number of workers being used by pytest-xdist
    '''
    vars_ = list(os.environ.keys())
    if 'PYTEST_XDIST_WORKER' in vars_ or 'PYTEST_XDIST_WORKER_COUNT' in vars_:
        return os.environ['PYTEST_XDIST_WORKER']
    else:
        return 'master'


def read_file_list() -> List[str]:
    '''Returns a list of paths to all files that currently exist in 'openml/tests/files/'

    :return: List[str]
    '''
    # TODO: better directory extractor
    static_cache_dir = os.getcwd()
    directory = os.path.join(static_cache_dir, 'tests/files/')
    if worker_id() == 'master':
        logger.info("Collecting file lists from: {}".format(directory))
    files = os.walk(directory)
    file_list = []
    for root, _, filenames in files:
        for filename in filenames:
            file_list.append(os.path.join(root, filename))
    return file_list


def compare_delete_files(old_list, new_list) -> None:
    '''Deletes files that are there in the new_list but not in the old_list

    :param old_list: List[str]
    :param new_list: List[str]
    :return: None
    '''
    file_list = list(set(new_list) - set(old_list))
    for file in file_list:
        os.remove(file)


def delete_remote_files(tracker) -> None:
    '''Function that deletes the entities passed as input, from the OpenML test server

    The TestBase class in openml/testing.py has an attribute called publish_tracker.
    This function expects the dictionary of the same structure.
    It is a dictionary of lists, where the keys are entity types, while the values are
    lists of integer IDs, except for key 'flow' where the value is a tuple (ID, flow name).

    Iteratively, multiple POST requests are made to the OpenML test server using
    openml.utils._delete_entity() to remove the entities uploaded by all the unit tests.

    :param tracker: Dict
    :return: None
    '''
    openml.config.server = TestBase.test_server
    openml.config.apikey = TestBase.apikey

    # legal_entities defined in openml.utils._delete_entity() - {'user'}
    # entity_types = {'run', 'data', 'flow', 'task', 'study'}
    # 'run' needs to be first entity to allow other dependent entities to be deleted

    # reordering to delete sub flows at the end of flows
    # sub-flows have shorter names, hence, sorting by descending order of flow name length
    if 'flow' in tracker:
        flow_deletion_order = [entity_id for entity_id, _ in
                               sorted(tracker['flow'], key=lambda x: len(x[1]), reverse=True)]
        tracker['flow'] = flow_deletion_order

    # deleting all collected entities published to test server
    # 'run's are deleted first to prevent dependency issue of entities on deletion
    logger.info("Entity Types: {}".format(['run', 'data', 'flow', 'task', 'study']))
    for entity_type in ['run', 'data', 'flow', 'task', 'study']:
        logger.info("Deleting {}s...".format(entity_type))
        for i, entity in enumerate(tracker[entity_type]):
            try:
                openml.utils._delete_entity(entity_type, entity)
                logger.info("Deleted ({}, {})".format(entity_type, entity))
            except Exception as e:
                logger.warn("Cannot delete ({},{}): {}".format(entity_type, entity, e))


def pytest_sessionstart() -> None:
    '''pytest hook that is executed before any unit test starts

    This function will be called by each of the worker processes, along with the master process
    when they are spawned. This happens even before the collection of unit tests.
    If number of workers, n=4, there will be a total of 5 (1 master + 4 workers) calls of this
    function, before execution of any unit test begins. The master pytest process has the name
    'master' while the worker processes are named as 'gw{i}' where i = 0, 1, ..., n-1.
    The order of process spawning is: 'master' -> random ordering of the 'gw{i}' workers.

    Since, master is always executed first, it is checked if the current process is 'master' and,
    * A semaphore is created which later will help synchronize the master and workers
    * Return a list of strings of paths of all files in the directory (pre-unit test snapshot)

    :return: None
    '''
    # file_list is global to maintain the directory snapshot during tear down
    global file_list
    worker = worker_id()
    if worker == 'master':
        # creates the semaphore which can be accessed using 'name'
        # initial_value is set to be 0
        # subsequently, a value of 0 would mean resource is occupied, 1 would mean it is available
        # for more details: http://semanchuk.com/philip/posix_ipc/#semaphore
        posix_ipc.Semaphore(name, flags=posix_ipc.O_CREAT, initial_value=0)
        file_list = read_file_list()
        # sets the semaphore to a value of 1, indicating it is available for other processes
        posix_ipc.Semaphore(name).release()
    logger.info("Start session: {}; Semaphore: {}".format(worker, posix_ipc.Semaphore(name).value))


def pytest_sessionfinish() -> None:
    '''pytest hook that is executed after all unit tests of a worker ends

    This function will be called by each of the worker processes, along with the master process
    when they are done with the unit tests allocated to them.
    If number of workers, n=4, there will be a total of 5 (1 master + 4 workers) calls of this
    function, before execution of any unit test begins. The master pytest process has the name
    'master' while the worker processes are named as 'gw{i}' where i = 0, 1, ..., n-1.
    The order of invocation is: random ordering of the 'gw{i}' workers -> 'master'.

    Since, master is always executed last, it is checked if the current process is 'master' and,
    * Compares file list with pre-unit test snapshot and deletes all local files generated
    * Reads the list of entities uploaded to test server and iteratively deletes them remotely
    * The semaphore is unlinked or deleted

    For the 'gw{i}' workers, this function:
    * Writes/updates a file which stores the dictionary containing the list of entities and their
      entity types that were uploaded to the test server by the unit tests
    The semaphore enforces synchronisation such that no parallel file read/write happens.
    The singular list of collated entity types allow a consistent deletion of all uploaded files,
    only after all unit tests have finished.

    :return: None
    '''
    # allows access to the file_list read in the set up phase
    global file_list
    worker = worker_id()
    logger.info("Finishing worker {}".format(worker))
    # locking - other workers go into 'wait' state, till the current worker calls 'release()'
    # this sets the semaphore value to 0, and hence, if any other worker has called 'acquire()'
    # in parallel, they enter a waiting queue, until the current process calls 'release()'
    posix_ipc.Semaphore(name).acquire()
    if worker == 'master':
        # Local file deletion
        new_file_list = read_file_list()
        compare_delete_files(file_list, new_file_list)
        logger.info("Local files deleted")

        # Test server file deletion
        #
        # Since master finished last, the file read now contains the collated list
        # from all the workers that were running in parallel
        with open(pkl_file, 'rb') as f:
            tracker = pickle.load(f)
        f.close()
        os.remove(pkl_file)
        delete_remote_files(tracker)
        logger.info("Remote files deleted")
        posix_ipc.Semaphore(name).release()
        logger.info("Master worker released")
        posix_ipc.unlink_semaphore(name)
        logger.info("Closed semaphore")
    else:  # If the process is a worker named 'gw{i}'
        if not os.path.isfile('publish_tracker.pkl'):
            # The first worker which has finished its allocated unit test will not find the
            # pickle file existing, and therefore will first create it
            with open(pkl_file, 'wb') as f:
                pickle.dump(TestBase.publish_tracker, f)
            f.close()
        # All workers that have finished their unit tests can read the pickle file
        with open(pkl_file, 'rb') as f:
            tracker = pickle.load(f)
        f.close()
        # 'tracker' collates the entity list from all workers into one
        for key in TestBase.publish_tracker:
            if key in tracker:
                tracker[key].extend(TestBase.publish_tracker[key])
                tracker[key] = list(set(tracker[key]))
            else:
                tracker[key] = TestBase.publish_tracker[key]
        # All workers finishing up, updates the pickle file
        with open(pkl_file, 'wb') as f:
            pickle.dump(tracker, f)
        f.close()
        logger.info("Releasing worker {}".format(worker))
        # The semaphore is made available for the other workers
        posix_ipc.Semaphore(name).release()
