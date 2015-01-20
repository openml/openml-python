__author__ = 'feurerm'

import unittest
import os
import shutil
import sys
import types

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
        self.connector = APIConnector(cache_directory=self.workdir)
        print(self.connector._session_hash)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.workdir)

    ############################################################################
    # Test administrative stuff
    @mock.patch.object(APIConnector, '_perform_api_call', autospec=True)
    def test_authentication(self, mock_perform_API_call):
        # TODO return error messages
        mock_perform_API_call.return_value = 400, \
        """<oml:authenticate xmlns:oml = "http://openml.org/openml">
  <oml:session_hash>G9MPPN114ZCZNWW2VN3JE9VF1FMV8Y5FXHUDUL4P</oml:session_hash>
  <oml:valid_until>2014-08-13 20:01:29</oml:valid_until>
  <oml:timezone>Europe/Berlin</oml:timezone>
</oml:authenticate>"""

        # This already does an authentication
        connector = APIConnector()
        # but we only test it here...
        self.assertEqual(1, mock_perform_API_call.call_count)
        self.assertEqual(connector._session_hash,
                         "G9MPPN114ZCZNWW2VN3JE9VF1FMV8Y5FXHUDUL4P")

        # Test that it actually returns what we want
        session_hash = connector._authenticate("Bla", "Blub")
        self.assertEqual(2, mock_perform_API_call.call_count)
        self.assertEqual(session_hash,
                         "G9MPPN114ZCZNWW2VN3JE9VF1FMV8Y5FXHUDUL4P")


    def test_parse_config(self):
        raise Exception()

    ############################################################################
    # Test all local stuff
    def test_get_cached_datasets(self):
        raise Exception()

    def test_get_cached_dataset(self):
        raise Exception()

    def test_get_cached_tasks(self):
        raise Exception()

    def test_get_cached_task(self):
        raise Exception()

    def test_get_cached_splits(self):
        raise Exception()

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


