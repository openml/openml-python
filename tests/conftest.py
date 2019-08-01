

import os
import pickle
import logging
import posix_ipc

import openml
from openml.testing import TestBase

# creating logger for unit test file deletion status
logger = logging.getLogger("unit_tests")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('UnitTestDeletion.log')
logger.addHandler(fh)

file_list = []
directory = None
name = '/test'
pkl_file = 'publish_tracker.pkl'


def worker_id():
    vars_ = list(os.environ.keys())
    if 'PYTEST_XDIST_WORKER' in vars_ or 'PYTEST_XDIST_WORKER_COUNT' in vars_:
        return os.environ['PYTEST_XDIST_WORKER']
    else:
        return 'master'


def read_file_list():
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


def compare_delete_files(old_list, new_list):
    file_list = list(set(new_list) - set(old_list))
    for file in file_list:
        os.remove(file)


def delete_remote_files(tracker):
    openml.config.server = TestBase.test_server
    openml.config.apikey = TestBase.apikey

    # legal_entities defined in openml.utils._delete_entity - {'user'}
    # entity_types = {'run', 'data', 'flow', 'task', 'study'}
    # 'run' needs to be first entity to allow other dependent entities to be deleted

    # reordering to delete sub flows at the end of flows
    # sub-flows have shorter names, hence, sorting by descending order of flow name length
    if 'flow' in tracker:
        flow_deletion_order = [entity_id for entity_id, _ in
                               sorted(tracker['flow'], key=lambda x: len(x[1]), reverse=True)]
        tracker['flow'] = flow_deletion_order

    # deleting all collected entities published to test server
    logger.info("Entity Types: {}".format(['run', 'data', 'flow', 'task', 'study']))
    for entity_type in ['run', 'data', 'flow', 'task', 'study']:
        logger.info("Deleting {}s...".format(entity_type))
        for i, entity in enumerate(tracker[entity_type]):
            try:
                # openml.utils._delete_entity(entity_type, entity)
                logger.info("Deleted ({}, {})".format(entity_type, entity))
                # deleting actual entry from tracker
                # TestBase._delete_entity_from_tracker(entity_type, entity)
            except Exception as e:
                logger.warn("Cannot delete ({},{}): {}".format(entity_type, entity, e))


#
# Pytest hooks
#


def pytest_sessionstart():
    global file_list
    worker = worker_id()
    if worker == 'master':
        posix_ipc.Semaphore(name, flags=posix_ipc.O_CREAT, initial_value=0)
        file_list = read_file_list()
        posix_ipc.Semaphore(name).release()
    logger.info("Start session: {}; Semaphore: {}".format(worker, posix_ipc.Semaphore(name).value))


def pytest_sessionfinish():
    global file_list
    worker = worker_id()
    logger.info("Finishing worker {}".format(worker))
    # locking - other workers go into 'wait' state, till the current worker calls 'release'
    posix_ipc.Semaphore(name).acquire()
    if worker == 'master':
        #
        # Local file deletion
        #
        new_file_list = read_file_list()
        compare_delete_files(file_list, new_file_list)
        logger.info("Local files deleted")
        #
        # Test server file deletion
        #
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
    else:
        # The first worker finishing up, creates the pickle file
        if not os.path.isfile('publish_tracker.pkl'):
            with open(pkl_file, 'wb') as f:
                pickle.dump(TestBase.publish_tracker, f)
            f.close()
        # All workers finishing up, reads the pickle file
        with open(pkl_file, 'rb') as f:
            tracker = pickle.load(f)
        f.close()
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
        posix_ipc.Semaphore(name).release()
