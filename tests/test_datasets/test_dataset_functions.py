import unittest
import os
import sys
import random
from itertools import product
if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import arff
import six

import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from oslo_concurrency import lockutils
from warnings import filterwarnings, catch_warnings

import openml
from openml import OpenMLDataset
from openml.exceptions import OpenMLCacheException, PyOpenMLError, \
    OpenMLHashException, PrivateDatasetError
from openml.testing import TestBase
from openml.utils import _tag_entity, _create_cache_directory_for_id
from openml.datasets.functions import (create_dataset,
                                       attributes_arff_from_df,
                                       _get_cached_dataset,
                                       _get_cached_dataset_features,
                                       _get_cached_dataset_qualities,
                                       _get_cached_datasets,
                                       _get_dataset_arff,
                                       _get_dataset_description,
                                       _get_dataset_features,
                                       _get_dataset_qualities,
                                       _get_online_dataset_arff,
                                       _get_online_dataset_format,
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

    def _get_empty_param_for_dataset(self):

        return {
            'name': None,
            'description': None,
            'creator': None,
            'contributor': None,
            'collection_date': None,
            'language': None,
            'licence': None,
            'default_target_attribute': None,
            'row_id_attribute': None,
            'ignore_attribute': None,
            'citation': None,
            'attributes': None,
            'data': None
        }

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
        did_cache_dir = _create_cache_directory_for_id(
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

        openml.datasets.get_dataset(3)
        file_path = os.path.join(openml.config.get_cache_directory(),
                                 "datasets", "3", "dataset.arff")
        dataset = OpenMLDataset(
            "anneal",
            "test",
            data_format="arff",
            version=1,
            licence="public",
            default_target_attribute="class",
            data_file=file_path,
        )
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
            "%s-UploadTestWithURL" % self._get_sentinel(),
            "test",
            data_format="arff",
            version=1,
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff",
        )
        dataset.publish()
        self.assertIsInstance(dataset.dataset_id, int)

    def test_data_status(self):
        dataset = OpenMLDataset(
            "%s-UploadTestWithURL" % self._get_sentinel(),
            "test", "ARFF",
            version=1,
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff")
        dataset.publish()
        did = dataset.dataset_id

        # admin key for test server (only adminds can activate datasets.
        # all users can deactivate their own datasets)
        openml.config.apikey = 'd488d8afd93b32331cf6ea9d7003d4c3'

        openml.datasets.status_update(did, 'active')
        # need to use listing fn, as this is immune to cache
        result = openml.datasets.list_datasets(data_id=did, status='all')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[did]['status'], 'active')
        openml.datasets.status_update(did, 'deactivated')
        # need to use listing fn, as this is immune to cache
        result = openml.datasets.list_datasets(data_id=did, status='all')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[did]['status'], 'deactivated')
        openml.datasets.status_update(did, 'active')
        # need to use listing fn, as this is immune to cache
        result = openml.datasets.list_datasets(data_id=did, status='all')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[did]['status'], 'active')
        with self.assertRaises(ValueError):
            openml.datasets.status_update(did, 'in_preparation')
        # need to use listing fn, as this is immune to cache
        result = openml.datasets.list_datasets(data_id=did, status='all')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[did]['status'], 'active')

    def test_attributes_arff_from_df(self):
        # DataFrame case
        df = pd.DataFrame(
            [[1, 1.0, 'xxx', 'A', True], [2, 2.0, 'yyy', 'B', False]],
            columns=['integer', 'floating', 'string', 'category', 'boolean']
        )
        df['category'] = df['category'].astype('category')
        attributes = attributes_arff_from_df(df)
        self.assertEqual(attributes, [('integer', 'INTEGER'),
                                      ('floating', 'REAL'),
                                      ('string', 'STRING'),
                                      ('category', ['A', 'B']),
                                      ('boolean', ['True', 'False'])])
        # SparseDataFrame case
        df = pd.SparseDataFrame([[1, 1.0],
                                 [2, 2.0],
                                 [0, 0]],
                                columns=['integer', 'floating'],
                                default_fill_value=0)
        df['integer'] = df['integer'].astype(np.int64)
        attributes = attributes_arff_from_df(df)
        self.assertEqual(attributes, [('integer', 'INTEGER'),
                                      ('floating', 'REAL')])

    def test_attributes_arff_from_df_mixed_dtype_categories(self):
        # liac-arff imposed categorical attributes to be of sting dtype. We
        # raise an error if this is not the case.
        df = pd.DataFrame([[1], ['2'], [3.]])
        df[0] = df[0].astype('category')
        err_msg = "The column '0' of the dataframe is of 'category' dtype."
        with pytest.raises(ValueError, match=err_msg):
            attributes_arff_from_df(df)

    def test_attributes_arff_from_df_unknown_dtype(self):
        # check that an error is raised when the dtype is not supported by
        # liac-arff
        data = [
            [[1], ['2'], [3.]],
            [pd.Timestamp('2012-05-01'), pd.Timestamp('2012-05-02')],
        ]
        dtype = [
            'mixed-integer',
            'datetime64'
        ]
        for arr, dt in zip(data, dtype):
            df = pd.DataFrame(arr)
            err_msg = ("The dtype '{}' of the column '0' is not currently "
                       "supported by liac-arff".format(dt))
            with pytest.raises(ValueError, match=err_msg):
                attributes_arff_from_df(df)

    def test_create_dataset_numpy(self):

        data = np.array(
            [
                [1, 2, 3],
                [1.2, 2.5, 3.8],
                [2, 5, 8],
                [0, 1, 0]
            ]
        ).T

        attributes = [('col_{}'.format(i), 'REAL')
                      for i in range(data.shape[1])]

        dataset = create_dataset(
            name='%s-NumPy_testing_dataset' % self._get_sentinel(),
            description='Synthetic dataset created from a NumPy array',
            creator='OpenML tester',
            contributor=None,
            collection_date='01-01-2018',
            language='English',
            licence='MIT',
            default_target_attribute='col_{}'.format(data.shape[1] - 1),
            row_id_attribute=None,
            ignore_attribute=None,
            citation='None',
            attributes=attributes,
            data=data,
            version_label='test',
            original_data_url='http://openml.github.io/openml-python',
            paper_url='http://openml.github.io/openml-python'
        )

        upload_did = dataset.publish()

        self.assertEqual(
            _get_online_dataset_arff(upload_did),
            dataset._dataset,
            "Uploaded arff does not match original one"
        )
        self.assertEqual(
            _get_online_dataset_format(upload_did),
            'arff',
            "Wrong format for dataset"
        )

    def test_create_dataset_list(self):

        data = [
            ['a', 'sunny', 85.0, 85.0, 'FALSE', 'no'],
            ['b', 'sunny', 80.0, 90.0, 'TRUE', 'no'],
            ['c', 'overcast', 83.0, 86.0, 'FALSE', 'yes'],
            ['d', 'rainy', 70.0, 96.0, 'FALSE', 'yes'],
            ['e', 'rainy', 68.0, 80.0, 'FALSE', 'yes'],
            ['f', 'rainy', 65.0, 70.0, 'TRUE', 'no'],
            ['g', 'overcast', 64.0, 65.0, 'TRUE', 'yes'],
            ['h', 'sunny', 72.0, 95.0, 'FALSE', 'no'],
            ['i', 'sunny', 69.0, 70.0, 'FALSE', 'yes'],
            ['j', 'rainy', 75.0, 80.0, 'FALSE', 'yes'],
            ['k', 'sunny', 75.0, 70.0, 'TRUE', 'yes'],
            ['l', 'overcast', 72.0, 90.0, 'TRUE', 'yes'],
            ['m', 'overcast', 81.0, 75.0, 'FALSE', 'yes'],
            ['n', 'rainy', 71.0, 91.0, 'TRUE', 'no'],
        ]

        attributes = [
            ('rnd_str', 'STRING'),
            ('outlook', ['sunny', 'overcast', 'rainy']),
            ('temperature', 'REAL'),
            ('humidity', 'REAL'),
            ('windy', ['TRUE', 'FALSE']),
            ('play', ['yes', 'no']),
        ]

        dataset = create_dataset(
            name="%s-ModifiedWeather" % self._get_sentinel(),
            description=(
                'Testing dataset upload when the data is a list of lists'
            ),
            creator='OpenML test',
            contributor=None,
            collection_date='21-09-2018',
            language='English',
            licence='MIT',
            default_target_attribute='play',
            row_id_attribute=None,
            ignore_attribute=None,
            citation='None',
            attributes=attributes,
            data=data,
            version_label='test',
            original_data_url='http://openml.github.io/openml-python',
            paper_url='http://openml.github.io/openml-python'
        )

        upload_did = dataset.publish()
        self.assertEqual(
            _get_online_dataset_arff(upload_did),
            dataset._dataset,
            "Uploaded ARFF does not match original one"
        )
        self.assertEqual(
            _get_online_dataset_format(upload_did),
            'arff',
            "Wrong format for dataset"
        )

    def test_create_dataset_sparse(self):

        # test the scipy.sparse.coo_matrix
        sparse_data = scipy.sparse.coo_matrix((
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ([0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1])
        ))

        column_names = [
            ('input1', 'REAL'),
            ('input2', 'REAL'),
            ('y', 'REAL'),
        ]

        xor_dataset = create_dataset(
            name="%s-XOR" % self._get_sentinel(),
            description='Dataset representing the XOR operation',
            creator=None,
            contributor=None,
            collection_date=None,
            language='English',
            licence=None,
            default_target_attribute='y',
            row_id_attribute=None,
            ignore_attribute=None,
            citation=None,
            attributes=column_names,
            data=sparse_data,
            version_label='test',
        )

        upload_did = xor_dataset.publish()
        self.assertEqual(
            _get_online_dataset_arff(upload_did),
            xor_dataset._dataset,
            "Uploaded ARFF does not match original one"
        )
        self.assertEqual(
            _get_online_dataset_format(upload_did),
            'sparse_arff',
            "Wrong format for dataset"
        )

        # test the list of dicts sparse representation
        sparse_data = [
            {0: 0.0},
            {1: 1.0, 2: 1.0},
            {0: 1.0, 2: 1.0},
            {0: 1.0, 1: 1.0}
        ]

        xor_dataset = create_dataset(
            name="%s-XOR" % self._get_sentinel(),
            description='Dataset representing the XOR operation',
            creator=None,
            contributor=None,
            collection_date=None,
            language='English',
            licence=None,
            default_target_attribute='y',
            row_id_attribute=None,
            ignore_attribute=None,
            citation=None,
            attributes=column_names,
            data=sparse_data,
            version_label='test',
        )

        upload_did = xor_dataset.publish()
        self.assertEqual(
            _get_online_dataset_arff(upload_did),
            xor_dataset._dataset,
            "Uploaded ARFF does not match original one"
        )
        self.assertEqual(
            _get_online_dataset_format(upload_did),
            'sparse_arff',
            "Wrong format for dataset"
        )

    def test_create_invalid_dataset(self):

        data = [
            'sunny',
            'overcast',
            'overcast',
            'rainy',
            'rainy',
            'rainy',
            'overcast',
            'sunny',
            'sunny',
            'rainy',
            'sunny',
            'overcast',
            'overcast',
            'rainy',
        ]

        param = self._get_empty_param_for_dataset()
        param['data'] = data

        self.assertRaises(
            ValueError,
            create_dataset,
            **param
        )

        param['data'] = data[0]
        self.assertRaises(
            ValueError,
            create_dataset,
            **param
        )

    def test_get_online_dataset_arff(self):

        # Australian dataset
        dataset_id = 100
        dataset = openml.datasets.get_dataset(dataset_id)
        decoder = arff.ArffDecoder()
        # check if the arff from the dataset is
        # the same as the arff from _get_arff function
        d_format = (dataset.format).lower()

        self.assertEqual(
            dataset._get_arff(d_format),
            decoder.decode(
                _get_online_dataset_arff(dataset_id),
                encode_nominal=True,
                return_type=arff.DENSE
                if d_format == 'arff' else arff.COO
            ),
            "ARFF files are not equal"
        )

    def test_get_online_dataset_format(self):

        # Phoneme dataset
        dataset_id = 77
        dataset = openml.datasets.get_dataset(dataset_id)

        self.assertEqual(
            (dataset.format).lower(),
            _get_online_dataset_format(dataset_id),
            "The format of the ARFF files is different"
        )

    def test_create_dataset_pandas(self):
        data = [
            ['a', 'sunny', 85.0, 85.0, 'FALSE', 'no'],
            ['b', 'sunny', 80.0, 90.0, 'TRUE', 'no'],
            ['c', 'overcast', 83.0, 86.0, 'FALSE', 'yes'],
            ['d', 'rainy', 70.0, 96.0, 'FALSE', 'yes'],
            ['e', 'rainy', 68.0, 80.0, 'FALSE', 'yes']
        ]
        column_names = ['rnd_str', 'outlook', 'temperature', 'humidity',
                        'windy', 'play']
        df = pd.DataFrame(data, columns=column_names)
        # enforce the type of each column
        df['outlook'] = df['outlook'].astype('category')
        df['windy'] = df['windy'].astype('bool')
        df['play'] = df['play'].astype('category')
        # meta-information
        name = '%s-pandas_testing_dataset' % self._get_sentinel()
        description = 'Synthetic dataset created from a Pandas DataFrame'
        creator = 'OpenML tester'
        collection_date = '01-01-2018'
        language = 'English'
        licence = 'MIT'
        default_target_attribute = 'play'
        citation = 'None'
        original_data_url = 'http://openml.github.io/openml-python'
        paper_url = 'http://openml.github.io/openml-python'
        dataset = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute=default_target_attribute,
            row_id_attribute=None,
            ignore_attribute=None,
            citation=citation,
            attributes='auto',
            data=df,
            version_label='test',
            original_data_url=original_data_url,
            paper_url=paper_url
        )
        upload_did = dataset.publish()
        self.assertEqual(
            _get_online_dataset_arff(upload_did),
            dataset._dataset,
            "Uploaded ARFF does not match original one"
        )

        # Check that SparseDataFrame are supported properly
        sparse_data = scipy.sparse.coo_matrix((
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ([0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1])
        ))
        column_names = ['input1', 'input2', 'y']
        df = pd.SparseDataFrame(sparse_data, columns=column_names)
        # meta-information
        description = 'Synthetic dataset created from a Pandas SparseDataFrame'
        dataset = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute=default_target_attribute,
            row_id_attribute=None,
            ignore_attribute=None,
            citation=citation,
            attributes='auto',
            data=df,
            version_label='test',
            original_data_url=original_data_url,
            paper_url=paper_url
        )
        upload_did = dataset.publish()
        self.assertEqual(
            _get_online_dataset_arff(upload_did),
            dataset._dataset,
            "Uploaded ARFF does not match original one"
        )
        self.assertEqual(
            _get_online_dataset_format(upload_did),
            'sparse_arff',
            "Wrong format for dataset"
        )

        # Check that we can overwrite the attributes
        data = [['a'], ['b'], ['c'], ['d'], ['e']]
        column_names = ['rnd_str']
        df = pd.DataFrame(data, columns=column_names)
        df['rnd_str'] = df['rnd_str'].astype('category')
        attributes = {'rnd_str': ['a', 'b', 'c', 'd', 'e', 'f', 'g']}
        dataset = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute=default_target_attribute,
            row_id_attribute=None,
            ignore_attribute=None,
            citation=citation,
            attributes=attributes,
            data=df,
            version_label='test',
            original_data_url=original_data_url,
            paper_url=paper_url
        )
        upload_did = dataset.publish()
        downloaded_data = _get_online_dataset_arff(upload_did)
        self.assertEqual(
            downloaded_data,
            dataset._dataset,
            "Uploaded ARFF does not match original one"
        )
        self.assertTrue(
            '@ATTRIBUTE rnd_str {a, b, c, d, e, f, g}' in downloaded_data)

    def test_create_dataset_row_id_attribute_error(self):
        # meta-information
        name = '%s-pandas_testing_dataset' % self._get_sentinel()
        description = 'Synthetic dataset created from a Pandas DataFrame'
        creator = 'OpenML tester'
        collection_date = '01-01-2018'
        language = 'English'
        licence = 'MIT'
        default_target_attribute = 'target'
        citation = 'None'
        original_data_url = 'http://openml.github.io/openml-python'
        paper_url = 'http://openml.github.io/openml-python'
        # Check that the index name is well inferred.
        data = [['a', 1, 0],
                ['b', 2, 1],
                ['c', 3, 0],
                ['d', 4, 1],
                ['e', 5, 0]]
        column_names = ['rnd_str', 'integer', 'target']
        df = pd.DataFrame(data, columns=column_names)
        # affecting row_id_attribute to an unknown column should raise an error
        err_msg = ("should be one of the data attribute.")
        with pytest.raises(ValueError, match=err_msg):
            openml.datasets.functions.create_dataset(
                name=name,
                description=description,
                creator=creator,
                contributor=None,
                collection_date=collection_date,
                language=language,
                licence=licence,
                default_target_attribute=default_target_attribute,
                ignore_attribute=None,
                citation=citation,
                attributes='auto',
                data=df,
                row_id_attribute='unknown_row_id',
                version_label='test',
                original_data_url=original_data_url,
                paper_url=paper_url
            )

    def test_create_dataset_row_id_attribute_inference(self):
        # meta-information
        name = '%s-pandas_testing_dataset' % self._get_sentinel()
        description = 'Synthetic dataset created from a Pandas DataFrame'
        creator = 'OpenML tester'
        collection_date = '01-01-2018'
        language = 'English'
        licence = 'MIT'
        default_target_attribute = 'target'
        citation = 'None'
        original_data_url = 'http://openml.github.io/openml-python'
        paper_url = 'http://openml.github.io/openml-python'
        # Check that the index name is well inferred.
        data = [['a', 1, 0],
                ['b', 2, 1],
                ['c', 3, 0],
                ['d', 4, 1],
                ['e', 5, 0]]
        column_names = ['rnd_str', 'integer', 'target']
        df = pd.DataFrame(data, columns=column_names)
        row_id_attr = [None, 'integer']
        df_index_name = [None, 'index_name']
        expected_row_id = [None, 'index_name', 'integer', 'integer']
        for output_row_id, (row_id, index_name) in zip(expected_row_id,
                                                       product(row_id_attr,
                                                               df_index_name)):
            df.index.name = index_name
            dataset = openml.datasets.functions.create_dataset(
                name=name,
                description=description,
                creator=creator,
                contributor=None,
                collection_date=collection_date,
                language=language,
                licence=licence,
                default_target_attribute=default_target_attribute,
                ignore_attribute=None,
                citation=citation,
                attributes='auto',
                data=df,
                row_id_attribute=row_id,
                version_label='test',
                original_data_url=original_data_url,
                paper_url=paper_url
            )
            self.assertEqual(dataset.row_id_attribute, output_row_id)
            upload_did = dataset.publish()
            arff_dataset = arff.loads(_get_online_dataset_arff(upload_did))
            arff_data = np.array(arff_dataset['data'], dtype=object)
            # if we set the name of the index then the index will be added to
            # the data
            expected_shape = (5, 3) if index_name is None else (5, 4)
            self.assertEqual(arff_data.shape, expected_shape)

    def test_create_dataset_attributes_auto_without_df(self):
        # attributes cannot be inferred without passing a dataframe
        data = np.array([[1, 2, 3],
                         [1.2, 2.5, 3.8],
                         [2, 5, 8],
                         [0, 1, 0]]).T
        attributes = 'auto'
        name = 'NumPy_testing_dataset'
        description = 'Synthetic dataset created from a NumPy array'
        creator = 'OpenML tester'
        collection_date = '01-01-2018'
        language = 'English'
        licence = 'MIT'
        default_target_attribute = 'col_{}'.format(data.shape[1] - 1)
        citation = 'None'
        original_data_url = 'http://openml.github.io/openml-python'
        paper_url = 'http://openml.github.io/openml-python'
        err_msg = "Automatically inferring the attributes required a pandas"
        with pytest.raises(ValueError, match=err_msg):
            openml.datasets.functions.create_dataset(
                name=name,
                description=description,
                creator=creator,
                contributor=None,
                collection_date=collection_date,
                language=language,
                licence=licence,
                default_target_attribute=default_target_attribute,
                row_id_attribute=None,
                ignore_attribute=None,
                citation=citation,
                attributes=attributes,
                data=data,
                version_label='test',
                original_data_url=original_data_url,
                paper_url=paper_url
            )
