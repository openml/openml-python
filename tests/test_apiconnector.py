__author__ = 'feurerm'

import unittest
import os
import shutil
import sys


if sys.version_info[0] >= 3:
    from unittest import mock
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from urllib.error import URLError
else:
    import mock
    from urllib import urlencode
    from urllib2 import URLError, urlopen


from openml.util import is_string

from openml import APIConnector
from openml import OpenMLDataset
from openml import OpenMLSplit


class TestAPIConnector(unittest.TestCase):
    """Test the APIConnector

    Note
    ----
    A config file with the username and password must be present to test the
    API calls.
    """

    def setUp(self):
        self.cwd = os.getcwd()
        workdir = os.path.dirname(os.path.abspath(__file__))
        self.workdir = os.path.join(workdir, "tmp")
        try:
            shutil.rmtree(self.workdir)
        except:
            pass

        os.mkdir(self.workdir)
        os.chdir(self.workdir)

        self.cached = True
        try:
            apikey = os.environ['OPENMLAPIKEY']
        except:
            apikey = None

        if "TRAVIS" in os.environ and apikey is None:
            raise Exception('Running on travis-ci, but no environment '
                            'variable OPENMLAPIKEY found.')

        self.connector = APIConnector(cache_directory=self.workdir,
                                      apikey=apikey)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.workdir)

    ############################################################################
    # Test administrative stuff
    @unittest.skip("Not implemented yet.")
    def test_parse_config(self):
        raise Exception()


    @unittest.skip("Not implemented yet.")
    def test_get_cached_tasks(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test_get_cached_task(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test_get_cached_splits(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test_get_cached_split(self):
        raise Exception()

    ############################################################################
    # Test all remote stuff


    ############################################################################
    # Tasks
    def test_get_task_list(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        def check_task(task):
            self.assertEqual(type(task), dict)
            self.assertGreaterEqual(len(task), 2)
            self.assertIn('did', task)
            self.assertIsInstance(task['did'], int)
            self.assertIn('status', task)
            self.assertTrue(is_string(task['status']))
            self.assertIn(task['status'],
                          ['in_preparation', 'active', 'deactivated'])

        tasks = self.connector.get_task_list(task_type_id=1)
        # 1759 as the number of supervised classification tasks retrieved
        # openml.org from this call; don't trust the number on openml.org as
        # it also counts private datasets
        self.assertGreaterEqual(len(tasks), 1759)
        for task in tasks:
            check_task(task)

        tasks = self.connector.get_task_list(task_type_id=2)
        self.assertGreaterEqual(len(tasks), 735)
        for task in tasks:
            check_task(task)

    def test_download_task(self):
        task = self.connector.download_task(1)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "task.xml")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "datasets", "1", "dataset.arff")))

    def test_download_split(self):
        task = self.connector.download_task(1)
        split = self.connector.download_split(task)
        self.assertEqual(type(split), OpenMLSplit)
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "tasks", "1", "datasplits.arff")))

    ############################################################################
    # Runs
    @unittest.skip('The method which is tested by this function doesnt exist')
    def test_download_run_list(self):
        def check_run(run):
            self.assertIsInstance(run, dict)
            self.assertEqual(len(run), 6)

        runs = self.connector.get_runs_list(task_id=1)
        self.assertGreaterEqual(len(runs), 800)
        for run in runs:
            check_run(run)

        runs = self.connector.get_runs_list(flow_id=1)
        self.assertGreaterEqual(len(runs), 1)
        for run in runs:
            check_run(run)

        runs = self.connector.get_runs_list(setup_id=1)
        self.assertGreaterEqual(len(runs), 260)
        for run in runs:
            check_run(run)

    def test_download_run(self):
        run = self.connector.download_run(473350)
        self.assertEqual(run.dataset_id, 1167)
        self.assertEqual(run.evaluations['f_measure'], 0.624668)

    # ###########################################################################
    # Flows
    @unittest.skip('The method which is tested by this function doesnt exist')
    def test_download_flow_list(self):
        def check_flow(flow):
            self.assertIsInstance(flow, dict)
            self.assertEqual(len(flow), 6)

        flows = self.connector.get_flow_list()
        self.assertGreaterEqual(len(flows), 1448)
        for flow in flows:
            check_flow(flow)


    def test_upload_flow(self):
        description = '''<oml:flow xmlns:oml="http://openml.org/openml"><oml:name>Test</oml:name><oml:description>description</oml:description> </oml:flow>'''
        return_code, dataset_xml = self.connector.upload_flow(description, "Testing upload flow")
        self.assertEqual(return_code, 200)

    def test_upload_run(self):
        url = urlopen("http://www.openml.org/data/download/224/weka_generated_predictions1977525485999711307.arff")
        prediction = url.read()

        description = '''<oml:run xmlns:oml="http://openml.org/openml"><oml:task_id>59</oml:task_id><oml:flow_id>67</oml:flow_id></oml:run>'''
        return_code, dataset_xml = self.connector.upload_run(prediction, description)
        self.assertEqual(return_code, 200)
