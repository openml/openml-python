import unittest
import os
import shutil
import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import openml
from openml import OpenMLDataset
from openml.exceptions import OpenMLCacheException
from openml.util import is_string
from openml.testing import TestBase

from openml.datasets.functions import (_get_cached_dataset,
                                       _get_cached_datasets,
                                       _get_dataset_description,
                                       _get_dataset_arff,
                                       _get_dataset_features,
                                       _get_dataset_qualities)


class TestOpenMLDataset(TestBase):

    def test__list_cached_datasets(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        cached_datasets = openml.datasets.functions._list_cached_datasets()
        self.assertIsInstance(cached_datasets, list)
        self.assertEqual(len(cached_datasets), 2)
        self.assertIsInstance(cached_datasets[0], int)

    @mock.patch('openml.datasets.functions._list_cached_datasets')
    def test__get_cached_datasets(self, _list_cached_datasets_mock):
        openml.config.set_cache_directory(self.static_cache_dir)
        _list_cached_datasets_mock.return_value = [-1, 2]
        datasets = _get_cached_datasets()
        self.assertIsInstance(datasets, dict)
        self.assertEqual(len(datasets), 2)
        self.assertIsInstance(list(datasets.values())[0], OpenMLDataset)

    def test__get_cached_dataset(self, ):
        openml.config.set_cache_directory(self.static_cache_dir)
        dataset = _get_cached_dataset(2)
        self.assertIsInstance(dataset, OpenMLDataset)

    def test_get_chached_dataset_description(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        description = openml.datasets.functions._get_cached_dataset_description(2)
        self.assertIsInstance(description, dict)

    def test_get_cached_dataset_description_not_cached(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        self.assertRaisesRegexp(OpenMLCacheException, "Dataset description for "
                                                      "did 3 not cached",
                                openml.datasets.functions._get_cached_dataset_description,
                                3)

    def test_get_cached_dataset_arff(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        description = openml.datasets.functions._get_cached_dataset_arff(2)
        self.assertIsInstance(description, str)

    def test_get_cached_dataset_arff_not_cached(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        self.assertRaisesRegexp(OpenMLCacheException, "ARFF file for "
                                                      "did 3 not cached",
                                openml.datasets.functions._get_cached_dataset_arff,
                                3)

    def test_list_datasets(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        datasets = openml.datasets.list_datasets()
        # 1087 as the number of datasets on openml.org
        self.assertGreaterEqual(len(datasets), 1087)
        for dataset in datasets:
            self.assertEqual(type(dataset), dict)
            self.assertGreaterEqual(len(dataset), 2)
            self.assertIn('did', dataset)
            self.assertIsInstance(dataset['did'], int)
            self.assertIn('status', dataset)
            self.assertTrue(is_string(dataset['status']))
            self.assertIn(dataset['status'], ['in_preparation', 'active',
                                              'deactivated'])

    def test_list_datasets_by_tag(self):
        datasets = openml.datasets.list_datasets_by_tag('uci')
        self.assertGreaterEqual(len(datasets), 5)
        for dataset in datasets:
            self.assertEqual(type(dataset), dict)
            self.assertGreaterEqual(len(dataset), 2)
            self.assertIn('did', dataset)
            self.assertIsInstance(dataset['did'], int)
            self.assertIn('status', dataset)
            self.assertTrue(is_string(dataset['status']))
            self.assertIn(dataset['status'], ['in_preparation', 'active',
                                              'deactivated'])

    def test_check_datasets_active(self):
        active = openml.datasets.check_datasets_active([1, 17])
        self.assertTrue(active[1])
        self.assertFalse(active[17])
        self.assertRaisesRegexp(ValueError, 'Could not find dataset 79 in OpenML'
                                            ' dataset list.',
                                openml.datasets.check_datasets_active, [79])

    def test_get_datasets(self):
        dids = [1, 2]
        datasets = openml.datasets.get_datasets(dids)
        self.assertEqual(len(datasets), 2)
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "2", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "dataset.arff")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "2", "dataset.arff")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "features.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "2", "features.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "qualities.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "2", "qualities.xml")))

    def test_get_dataset(self):
        dataset = openml.datasets.get_dataset(1)
        self.assertEqual(type(dataset), OpenMLDataset)
        self.assertEqual(dataset.name, 'anneal')
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "description.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "dataset.arff")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "features.xml")))
        self.assertTrue(os.path.exists(os.path.join(
            openml.config.get_cache_directory(), "datasets", "1", "qualities.xml")))

    def test_download_rowid(self):
        # Smoke test which checks that the dataset has the row-id set correctly
        did = 164
        dataset = openml.datasets.get_dataset(did)
        self.assertEqual(dataset.row_id_attribute, 'instance')

    def test__get_dataset_description(self):
        description = _get_dataset_description(self.workdir, 2)
        self.assertIsInstance(description, dict)
        description_xml_path = os.path.join(self.workdir,
                                            'description.xml')
        self.assertTrue(os.path.exists(description_xml_path))

    def test__getarff_path_dataset_arff(self):
        openml.config.set_cache_directory(self.static_cache_dir)
        description = openml.datasets.functions._get_cached_dataset_description(2)
        arff_path = _get_dataset_arff(self.workdir, description)
        self.assertIsInstance(arff_path, str)
        self.assertTrue(os.path.exists(arff_path))

    def test__get_dataset_features(self):
        features = _get_dataset_features(self.workdir, 2)
        self.assertIsInstance(features, dict)
        features_xml_path = os.path.join(self.workdir, 'features.xml')
        self.assertTrue(os.path.exists(features_xml_path))

    def test__get_dataset_qualities(self):
        # Only a smoke check
        qualities = _get_dataset_qualities(self.workdir, 2)
        self.assertIsInstance(qualities, dict)

    def test_deletion_of_cache_dir(self):
        # Simple removal
        did_cache_dir = openml.datasets.functions.\
            _create_dataset_cache_directory(1)
        self.assertTrue(os.path.exists(did_cache_dir))
        openml.datasets.functions._remove_dataset_cache_dir(did_cache_dir)
        self.assertFalse(os.path.exists(did_cache_dir))

    # Use _get_dataset_arff to load the description, trigger an exception in the
    # test target and have a slightly higher coverage
    @mock.patch('openml.datasets.functions._get_dataset_arff')
    def test_deletion_of_cache_dir_faulty_download(self, patch):
        patch.side_effect = Exception('Boom!')
        self.assertRaisesRegexp(Exception, 'Boom!', openml.datasets.get_dataset,
                                1)
        datasets_cache_dir = os.path.join(self.workdir, 'datasets')
        self.assertEqual(len(os.listdir(datasets_cache_dir)), 0)

    def test_publish_dataset(self):
        dataset = openml.datasets.get_dataset(3)
        file_path = os.path.join(openml.config.get_cache_directory(),
                                 "datasets", "3", "dataset.arff")
        dataset = OpenMLDataset(
            name="anneal", version=1, description="test",
            format="ARFF", licence="public", default_target_attribute="class", data_file=file_path)
        return_code, return_value = dataset.publish()
        self.assertEqual(return_code, 200)

    def test_upload_dataset_with_url(self):
        dataset = OpenMLDataset(
            name="UploadTestWithURL", version=1, description="test",
            format="ARFF",
            url="http://expdb.cs.kuleuven.be/expdb/data/uci/nominal/iris.arff")
        return_code, return_value = dataset.publish()
        # self.assertTrue("This is a read-only account" in return_value)
        self.assertEqual(return_code, 200)
