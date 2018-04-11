import unittest
import os
import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock


import random
import six

from oslo_concurrency import lockutils

import scipy.sparse

import openml
from openml import OpenMLDataset
from openml.exceptions import OpenMLCacheException, PyOpenMLError, \
    OpenMLHashException, PrivateDatasetError
from openml.testing import TestBase
from openml.utils import _tag_entity, _create_cache_directory_for_id

from openml.datasets.functions import (_get_cached_dataset,
                                       _get_cached_dataset_features,
                                       _get_cached_dataset_qualities,
                                       _get_cached_datasets,
                                       _get_dataset_description,
                                       _get_dataset_arff,
                                       _get_dataset_features,
                                       _get_dataset_qualities,
                                       DATASETS_CACHE_DIR_NAME)


class TestOpenMLDataset(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super(TestOpenMLDataset, self).setUp()

    def tearDown(self):
        self._remove_pickle_files()
        super(TestOpenMLDataset, self).tearDown()

    def _remove_pickle_files(self):
        cache_dir = self.static_cache_dir
        for did in ['-1', '2']:
            with lockutils.external_lock(
                    name='datasets.functions.get_dataset:%s' % did,
                    lock_path=os.path.join(openml.config.get_cache_directory(), 'locks'),
            ):
                pickle_path = os.path.join(cache_dir, 'datasets', did,
                                           'dataset.pkl')
                try:
                    os.remove(pickle_path)
                except:
                    pass

    def test__list_cached_datasets(self):
        openml.config.cache_directory = self.static_cache_dir
        cached_datasets = openml.datasets.functions._list_cached_datasets()
        self.assertIsInstance(cached_datasets, list)
        self.assertEqual(len(cached_datasets), 2)
        self.assertIsInstance(cached_datasets[0], int)

    @mock.patch('openml.datasets.functions._list_cached_datasets')
    def test__get_cached_datasets(self, _list_cached_datasets_mock):
        openml.config.cache_directory = self.static_cache_dir
        _list_cached_datasets_mock.return_value = [-1, 2]
        datasets = _get_cached_datasets()
        self.assertIsInstance(datasets, dict)
        self.assertEqual(len(datasets), 2)
        self.assertIsInstance(list(datasets.values())[0], OpenMLDataset)

    def test__get_cached_dataset(self, ):
        openml.config.cache_directory = self.static_cache_dir
        dataset = _get_cached_dataset(2)
        features = _get_cached_dataset_features(2)
        qualities = _get_cached_dataset_qualities(2)
        self.assertIsInstance(dataset, OpenMLDataset)
        self.assertTrue(len(dataset.features) > 0)
        self.assertTrue(len(dataset.features) == len(features['oml:feature']))
        self.assertTrue(len(dataset.qualities) == len(qualities))

    def test_get_cached_dataset_description(self):
        openml.config.cache_directory = self.static_cache_dir
        description = openml.datasets.functions._get_cached_dataset_description(2)
        self.assertIsInstance(description, dict)

    def test_get_cached_dataset_description_not_cached(self):
        openml.config.cache_directory = self.static_cache_dir
        self.assertRaisesRegexp(OpenMLCacheException, "Dataset description for "
                                                      "dataset id 3 not cached",
                                openml.datasets.functions._get_cached_dataset_description,
                                3)

    def test_get_cached_dataset_arff(self):
        openml.config.cache_directory = self.static_cache_dir
        description = openml.datasets.functions._get_cached_dataset_arff(
            dataset_id=2)
        self.assertIsInstance(description, str)

    def test_get_cached_dataset_arff_not_cached(self):
        openml.config.cache_directory = self.static_cache_dir
        self.assertRaisesRegexp(OpenMLCacheException, "ARFF file for "
                                                      "dataset id 3 not cached",
                                openml.datasets.functions._get_cached_dataset_arff,
                                3)

    def _check_dataset(self, dataset):
            self.assertEqual(type(dataset), dict)
            self.assertGreaterEqual(len(dataset), 2)
            self.assertIn('did', dataset)
            self.assertIsInstance(dataset['did'], int)
            self.assertIn('status', dataset)
            self.assertIsInstance(dataset['status'], six.string_types)
            self.assertIn(dataset['status'], ['in_preparation', 'active',
                                              'deactivated'])
    def _check_datasets(self, datasets):
        for did in datasets:
            self._check_dataset(datasets[did])

    def test_tag_untag_dataset(self):
        tag = 'test_tag_%d' %random.randint(1, 1000000)
        all_tags = _tag_entity('data', 1, tag)
        self.assertTrue(tag in all_tags)
        all_tags = _tag_entity('data', 1, tag, untag=True)
        self.assertTrue(tag not in all_tags)

    def test_list_datasets(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        datasets = openml.datasets.list_datasets()
        # 1087 as the number of datasets on openml.org
        self.assertGreaterEqual(len(datasets), 100)
        self._check_datasets(datasets)

    def test_list_datasets_by_tag(self):
        datasets = openml.datasets.list_datasets(tag='study_14')
        self.assertGreaterEqual(len(datasets), 100)
        self._check_datasets(datasets)

    def test_list_datasets_by_size(self):
        datasets = openml.datasets.list_datasets(size=10050)
        self.assertGreaterEqual(len(datasets), 120)
        self._check_datasets(datasets)

    def test_list_datasets_by_number_instances(self):
        datasets = openml.datasets.list_datasets(number_instances="5..100")
        self.assertGreaterEqual(len(datasets), 4)
        self._check_datasets(datasets)

    def test_list_datasets_by_number_features(self):
        datasets = openml.datasets.list_datasets(number_features="50..100")
        self.assertGreaterEqual(len(datasets), 8)
        self._check_datasets(datasets)

    def test_list_datasets_by_number_classes(self):
        datasets = openml.datasets.list_datasets(number_classes="5")
        self.assertGreaterEqual(len(datasets), 3)
        self._check_datasets(datasets)

    def test_list_datasets_by_number_missing_values(self):
        datasets = openml.datasets.list_datasets(number_missing_values="5..100")
        self.assertGreaterEqual(len(datasets), 5)
        self._check_datasets(datasets)

    def test_list_datasets_combined_filters(self):
        datasets = openml.datasets.list_datasets(tag='study_14', number_instances="100..1000", number_missing_values="800..1000")
        self.assertGreaterEqual(len(datasets), 1)
        self._check_datasets(datasets)

    def test_list_datasets_paginate(self):
        size = 10
        max = 100
        for i in range(0, max, size):
            datasets = openml.datasets.list_datasets(offset=i, size=size)
            self.assertEqual(size, len(datasets))
            self._check_datasets(datasets)

    def test_list_datasets_empty(self):
        datasets = openml.datasets.list_datasets(tag='NoOneWouldUseThisTagAnyway')
        if len(datasets) > 0:
            raise ValueError('UnitTest Outdated, tag was already used (please remove)')

        self.assertIsInstance(datasets, dict)

    @unittest.skip('See https://github.com/openml/openml-python/issues/149')
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

        self.assertGreater(len(dataset.features), 1)
        self.assertGreater(len(dataset.qualities), 4)

        # Issue324 Properly handle private datasets when trying to access them
        openml.config.server = self.production_server
        self.assertRaises(PrivateDatasetError, openml.datasets.get_dataset, 45)


    def test_get_dataset_with_string(self):
        dataset = openml.datasets.get_dataset(101)
        self.assertRaises(PyOpenMLError, dataset._get_arff, 'arff')
        self.assertRaises(PyOpenMLError, dataset.get_data)

    def test_get_dataset_sparse(self):
        dataset = openml.datasets.get_dataset(102)
        X = dataset.get_data()
        self.assertIsInstance(X, scipy.sparse.csr_matrix)

    def test_download_rowid(self):
        # Smoke test which checks that the dataset has the row-id set correctly
        did = 44
        dataset = openml.datasets.get_dataset(did)
        self.assertEqual(dataset.row_id_attribute, 'Counter')

    def test__get_dataset_description(self):
        description = _get_dataset_description(self.workdir, 2)
        self.assertIsInstance(description, dict)
        description_xml_path = os.path.join(self.workdir,
                                            'description.xml')
        self.assertTrue(os.path.exists(description_xml_path))

    def test__getarff_path_dataset_arff(self):
        openml.config.cache_directory = self.static_cache_dir
        description = openml.datasets.functions._get_cached_dataset_description(2)
        arff_path = _get_dataset_arff(self.workdir, description)
        self.assertIsInstance(arff_path, str)
        self.assertTrue(os.path.exists(arff_path))

    def test__getarff_md5_issue(self):
        description = {
            'oml:id': 5,
            'oml:md5_checksum': 'abc',
            'oml:url': 'https://www.openml.org/data/download/61',
        }
        self.assertRaisesRegexp(
            OpenMLHashException,
            'Checksum ad484452702105cbf3d30f8deaba39a9 of downloaded dataset 5 '
            'is unequal to the checksum abc sent by the server.',
            _get_dataset_arff,
            self.workdir, description,
        )

    def test__get_dataset_features(self):
        features = _get_dataset_features(self.workdir, 2)
        self.assertIsInstance(features, dict)
        features_xml_path = os.path.join(self.workdir, 'features.xml')
        self.assertTrue(os.path.exists(features_xml_path))

    def test__get_dataset_qualities(self):
        # Only a smoke check
        qualities = _get_dataset_qualities(self.workdir, 2)
        self.assertIsInstance(qualities, list)

    def test_deletion_of_cache_dir(self):
        # Simple removal
        did_cache_dir = openml.utils._create_cache_directory_for_id(
            DATASETS_CACHE_DIR_NAME, 1,
        )
        self.assertTrue(os.path.exists(did_cache_dir))
        openml.utils._remove_cache_dir_for_id(
            DATASETS_CACHE_DIR_NAME, did_cache_dir,
        )
        self.assertFalse(os.path.exists(did_cache_dir))

    # Use _get_dataset_arff to load the description, trigger an exception in the
    # test target and have a slightly higher coverage
    @mock.patch('openml.datasets.functions._get_dataset_arff')
    def test_deletion_of_cache_dir_faulty_download(self, patch):
        patch.side_effect = Exception('Boom!')
        self.assertRaisesRegexp(Exception, 'Boom!', openml.datasets.get_dataset,
                                1)
        datasets_cache_dir = os.path.join(
            self.workdir, 'org', 'openml', 'test', 'datasets'
        )
        self.assertEqual(len(os.listdir(datasets_cache_dir)), 0)

    def test_publish_dataset(self):
        dataset = openml.datasets.get_dataset(3)
        file_path = os.path.join(openml.config.get_cache_directory(),
                                 "datasets", "3", "dataset.arff")
        dataset = OpenMLDataset(
            name="anneal", version=1, description="test",
            format="ARFF", licence="public", default_target_attribute="class", data_file=file_path)
        dataset.publish()
        self.assertIsInstance(dataset.dataset_id, int)

    def test__retrieve_class_labels(self):
        openml.config.cache_directory = self.static_cache_dir
        labels = openml.datasets.get_dataset(2).retrieve_class_labels()
        self.assertEqual(labels, ['1', '2', '3', '4', '5', 'U'])
        labels = openml.datasets.get_dataset(2).retrieve_class_labels(
            target_name='product-type')
        self.assertEqual(labels, ['C', 'H', 'G'])

    def test_upload_dataset_with_url(self):
        dataset = OpenMLDataset(
            name="UploadTestWithURL", version=1, description="test",
            format="ARFF",
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff")
        dataset.publish()
        self.assertIsInstance(dataset.dataset_id, int)
