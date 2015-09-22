__author__ = 'feurerm'

import unittest
import os
import shutil
import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

from openml.util import is_string

from openml.apiconnector import APIConnector
from openml.entities.dataset import OpenMLDataset
from openml.entities.split import OpenMLSplit


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

    ############################################################################
    # Test all local stuff
    def test_get_cached_datasets(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")
        connector = APIConnector(cache_directory=workdir)
        datasets = connector.get_cached_datasets()
        self.assertIsInstance(datasets, dict)
        self.assertEqual(len(datasets), 2)
        self.assertIsInstance(list(datasets.values())[0], OpenMLDataset)

    def test_get_cached_dataset(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")

        with mock.patch.object(APIConnector, "_perform_api_call") as api_mock:
            api_mock.return_value = 400, \
                """<oml:authenticate xmlns:oml = "http://openml.org/openml">
                <oml:session_hash>G9MPPN114ZCZNWW2VN3JE9VF1FMV8Y5FXHUDUL4P</oml:session_hash>
                <oml:valid_until>2014-08-13 20:01:29</oml:valid_until>
                <oml:timezone>Europe/Berlin</oml:timezone>
                </oml:authenticate>"""

            connector = APIConnector(cache_directory=workdir)
            dataset = connector.get_cached_dataset(2)
            self.assertIsInstance(dataset, OpenMLDataset)
            self.assertTrue(connector._perform_api_call.is_called_once())

    def test_get_chached_dataset_description(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")
        connector = APIConnector(cache_directory=workdir)
        description = connector._get_cached_dataset_description(2)
        self.assertIsInstance(description, dict)

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
    # Datasets
    def test_get_dataset_list(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        datasets = self.connector.get_dataset_list()
        # 1087 as the number of datasets on openml.org
        self.assertTrue(len(datasets) >= 1087)
        for dataset in datasets:
            self.assertEqual(type(dataset), dict)
            self.assertGreaterEqual(len(dataset), 2)
            self.assertIn('did', dataset)
            self.assertIsInstance(dataset['did'], int)
            self.assertIn('status', dataset)
            self.assertTrue(is_string(dataset['status']))
            self.assertIn(dataset['status'], ['in_preparation', 'active',
                                              'deactivated'])

    @unittest.skip("Not implemented yet.")
    def test_datasets_active(self):
        raise NotImplementedError()

    def test_download_datasets(self):
        dids = [1, 2]
        datasets = self.connector.download_datasets(dids)
        self.assertEqual(len(datasets), 2)
        self.assertTrue(os.path.exists(os.path.join(
            self.connector.dataset_cache_dir, "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            self.connector.dataset_cache_dir, "2", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            self.connector.dataset_cache_dir, "1", "dataset.arff")))
        self.assertTrue(os.path.exists(os.path.join(
            self.connector.dataset_cache_dir, "2", "dataset.arff")))

    def test_download_dataset(self):
        dataset = self.connector.download_dataset(1)
        self.assertEqual(type(dataset), OpenMLDataset)
        self.assertEqual(dataset.name, 'anneal')
        self.assertTrue(os.path.exists(os.path.join(
            self.connector.dataset_cache_dir, "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            self.connector.dataset_cache_dir, "1", "dataset.arff")))

    def test_download_rowid(self):
        # Smoke test which checks that the dataset has the row-id set correctly
        did = 164
        dataset = self.connector.download_dataset(did)
        self.assertEqual(dataset.row_id_attribute, 'instance')

    def test_download_dataset_description(self):
        # Only a smoke test, I don't know exactly how to test the URL
        # retrieval and "caching"
        description = self.connector.download_dataset_description(2)
        self.assertIsInstance(description, dict)

    def test_download_dataset_features(self):
        # Only a smoke check
        features = self.connector.download_dataset_features(2)
        self.assertIsInstance(features, dict)

    def test_download_dataset_qualities(self):
        # Only a smoke check
        qualities = self.connector.download_dataset_qualities(2)
        self.assertIsInstance(qualities, dict)

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
        self.assertGreaterEqual(len(run.tags), 2)
        self.assertEqual(len(run.datasets), 1)
        self.assertGreaterEqual(len(run.files), 2)
        self.assertGreaterEqual(len(run.evaluations), 18)
        self.assertEqual(len(run.evaluations['f_measure']), 2)

    # ###########################################################################
    # Flows
    def test_download_flow_list(self):
        def check_flow(flow):
            self.assertIsInstance(flow, dict)
            self.assertEqual(len(flow), 6)

        flows = self.connector.get_flow_list()
        self.assertGreaterEqual(len(flows), 1448)
        for flow in flows:
            check_flow(flow)

    def test_upload_dataset(self):

        dataset = self.connector.download_dataset(3)
        filePath = os.path.join(self.connector.dataset_cache_dir, "3", "dataset.arff")

        description = """ <oml:data_set_description xmlns:oml="http://openml.org/openml">
                        <oml:name>anneal</oml:name>
                        <oml:version>1</oml:version>
                        <oml:description>test</oml:description>
                        <oml:format>ARFF</oml:format>
                        <oml:licence>Public</oml:licence>
                        <oml:default_target_attribute>class</oml:default_target_attribute>
                        <oml:md5_checksum></oml:md5_checksum>
                        </oml:data_set_description>
                         """
        return_code, dataset_xml = self.connector.upload_dataset(description, filePath)
        self.assertEqual(return_code, 200)

    def test_upload_dataset_with_url(self):

        description = """ <oml:data_set_description xmlns:oml="http://openml.org/openml">
                        <oml:name>UploadTestWithURL</oml:name>
                        <oml:version>1</oml:version>
                        <oml:description>test</oml:description>
                        <oml:format>ARFF</oml:format>
                        <oml:url>http://expdb.cs.kuleuven.be/expdb/data/uci/nominal/iris.arff</oml:url>
                        </oml:data_set_description>
                         """
        return_code, dataset_xml = self.connector.upload_dataset (description)
        self.assertEqual(return_code, 200)

    def test_upload_flow(self):

        description = """ <oml:data_set_description xmlns:oml="http://openml.org/openml">
                        <oml:name>UploadTestWithURL</oml:name>
                        <oml:version>1</oml:version>
                        <oml:description>test</oml:description>
                        <oml:format>ARFF</oml:format>
                        <oml:url>http://expdb.cs.kuleuven.be/expdb/data/uci/nominal/iris.arff</oml:url>
                        </oml:data_set_description>
                         """
        return_code, dataset_xml = self.connector.upload_dataset (description)
        self.assertEqual(return_code, 200)






