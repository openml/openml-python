import unittest
import os
import sys
import shutil

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

from openml import APIConnector
from openml import OpenMLDataset
from openml.util import is_string
import openml


class TestOpenMLDataset(unittest.TestCase):

    def setUp(self):
        # FIXME REFACTOR with test apiconnector
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

        self.con = APIConnector(cache_directory=self.workdir, apikey=apikey)

    ############################################################################
    # Test all local stuff
    def test_get_cached_datasets(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")
        con = APIConnector(cache_directory=workdir)
        datasets = openml.datasets.get_cached_datasets(con)
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
            dataset = openml.datasets.get_cached_dataset(connector, 2)
            self.assertIsInstance(dataset, OpenMLDataset)
            self.assertTrue(connector._perform_api_call.is_called_once())

    def test_get_chached_dataset_description(self):
        workdir = os.path.dirname(os.path.abspath(__file__))
        workdir = os.path.join(workdir, "files")
        connector = APIConnector(cache_directory=workdir)
        description = openml.datasets.functions._get_cached_dataset_description(connector,
                                                                      2)
        self.assertIsInstance(description, dict)

    ############################################################################
    # Datasets
    def test_get_dataset_list(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        datasets = openml.datasets.get_dataset_list(self.con)
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
        datasets = openml.datasets.download_datasets(self.con, dids)
        self.assertEqual(len(datasets), 2)
        self.assertTrue(os.path.exists(os.path.join(
            self.con.dataset_cache_dir, "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            self.con.dataset_cache_dir, "2", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            self.con.dataset_cache_dir, "1", "dataset.arff")))
        self.assertTrue(os.path.exists(os.path.join(
            self.con.dataset_cache_dir, "2", "dataset.arff")))

    def test_download_dataset(self):
        dataset = openml.datasets.download_dataset(self.con, 1)
        self.assertEqual(type(dataset), OpenMLDataset)
        self.assertEqual(dataset.name, 'anneal')
        self.assertTrue(os.path.exists(os.path.join(
            self.con.dataset_cache_dir, "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            self.con.dataset_cache_dir, "1", "dataset.arff")))

    def test_download_rowid(self):
        # Smoke test which checks that the dataset has the row-id set correctly
        did = 164
        dataset = openml.datasets.download_dataset(self.con, did)
        self.assertEqual(dataset.row_id_attribute, 'instance')

    def test_download_dataset_description(self):
        # Only a smoke test, I don't know exactly how to test the URL
        # retrieval and "caching"
        description = openml.datasets.download_dataset_description(self.con, 2)
        self.assertIsInstance(description, dict)

    def test_download_dataset_features(self):
        # Only a smoke check
        features = openml.datasets.download_dataset_features(self.con, 2)
        self.assertIsInstance(features, dict)

    def test_download_dataset_qualities(self):
        # Only a smoke check
        qualities = openml.datasets.download_dataset_qualities(self.con, 2)
        self.assertIsInstance(qualities, dict)

    def test_upload_dataset(self):

        dataset = openml.datasets.download_dataset(self.con, 3)
        file_path = os.path.join(self.connector.dataset_cache_dir, "3", "dataset.arff")

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
        return_code, dataset_xml = openml.datasets.upload_dataset(self.con, description, file_path)
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
        return_code, dataset_xml = openml.datasets.upload_dataset(self.con, description)
        self.assertEqual(return_code, 200)
