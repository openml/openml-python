import hashlib
import inspect
import os
import shutil
import time
import unittest

from oslo_concurrency import lockutils
import six

import openml


class TestBase(unittest.TestCase):
    """Base class for tests

    Note
    ----
    Currently hard-codes a read-write key.
    Hopefully soon allows using a test server, not the production server.
    """

    def setUp(self):
        # This cache directory is checked in to git to simulate a populated
        # cache
        self.maxDiff = None
        self.static_cache_dir = None
        static_cache_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
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
        try:
            shutil.rmtree(self.workdir)
        except:
            pass

        os.mkdir(self.workdir)
        os.chdir(self.workdir)

        self.cached = True
        # amueller's read/write key that he will throw away later
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        self.production_server = "https://openml.org/api/v1/xml"
        self.test_server = "https://test.openml.org/api/v1/xml"
        openml.config.cache_directory = None

        openml.config.server = self.test_server
        openml.config.avoid_duplicate_runs = False

        openml.config.cache_directory = self.workdir

        # If we're on travis, we save the api key in the config file to allow
        # the notebook tests to read them.
        if os.environ.get('TRAVIS') or os.environ.get('APPVEYOR'):
            with lockutils.external_lock('config', lock_path=self.workdir):
                with open(openml.config.config_file, 'w') as fh:
                    fh.write('apikey = %s' % openml.config.apikey)

        # Increase the number of retries to avoid spurios server failures
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

    def _get_sentinel(self, sentinel=None):
        if sentinel is None:
            # Create a unique prefix for the flow. Necessary because the flow is
            # identified by its name and external version online. Having a unique
            #  name allows us to publish the same flow in each test run
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
        self.assertIsInstance(dataset['status'], six.string_types)
        self.assertIn(dataset['status'], ['in_preparation', 'active',
                                          'deactivated'])


__all__ = ['TestBase']
