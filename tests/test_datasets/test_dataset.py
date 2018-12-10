from time import time, sleep
from warnings import filterwarnings, catch_warnings

import six
import numpy as np
import pandas as pd
from scipy import sparse

import openml
from openml.testing import TestBase
from openml.datasets.functions import _get_online_dataset_arff
from openml.datasets.functions import attributes_arff_from_df
from openml.datasets import get_dataset
from openml.exceptions import OpenMLServerException


class OpenMLDatasetTest(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super(OpenMLDatasetTest, self).setUp()
        openml.config.server = self.production_server

        # Load dataset id 2 - dataset 2 is interesting because it contains
        # missing values, categorical features etc.
        self.dataset = openml.datasets.get_dataset(2)
        self.titanic = openml.datasets.get_dataset(40945)

    def test_get_data(self):
        # Basic usage
        rval = self.dataset.get_data()
        self.assertIsInstance(rval, np.ndarray)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual((898, 39), rval.shape)
        rval, categorical = self.dataset.get_data(
            return_categorical_indicator=True)
        self.assertEqual(len(categorical), 39)
        self.assertTrue(all([isinstance(cat, bool) for cat in categorical]))
        rval, attribute_names = self.dataset.get_data(
            return_attribute_names=True)
        self.assertEqual(len(attribute_names), 39)
        self.assertTrue(all([isinstance(att, six.string_types)
                             for att in attribute_names]))

    def test_get_data_with_rowid(self):
        self.dataset.row_id_attribute = "condition"
        rval, categorical = self.dataset.get_data(
            include_row_id=True, return_categorical_indicator=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 39))
        self.assertEqual(len(categorical), 39)
        rval, categorical = self.dataset.get_data(
            include_row_id=False, return_categorical_indicator=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 38))
        self.assertEqual(len(categorical), 38)

    def test_get_data_with_target(self):
        X, y = self.dataset.get_data(target="class")
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.dtype, np.float32)
        self.assertIn(y.dtype, [np.int32, np.int64])
        self.assertEqual(X.shape, (898, 38))
        X, y, attribute_names = self.dataset.get_data(
            target="class",
            return_attribute_names=True
        )
        self.assertEqual(len(attribute_names), 38)
        self.assertNotIn("class", attribute_names)
        self.assertEqual(y.shape, (898, ))

    def test_get_data_rowid_and_ignore_and_target(self):
        self.dataset.ignore_attributes = ["condition"]
        self.dataset.row_id_attribute = ["hardness"]
        X, y = self.dataset.get_data(
            target="class",
            include_row_id=False,
            include_ignore_attributes=False
        )
        self.assertEqual(X.dtype, np.float32)
        self.assertIn(y.dtype, [np.int32, np.int64])
        self.assertEqual(X.shape, (898, 36))
        X, y, categorical = self.dataset.get_data(
            target="class",
            return_categorical_indicator=True,
        )
        self.assertEqual(len(categorical), 36)
        self.assertListEqual(categorical, [True] * 3 + [False] + [True] * 2 + [
            False] + [True] * 23 + [False] * 3 + [True] * 3)
        self.assertEqual(y.shape, (898, ))

    def test_get_data_with_ignore_attributes(self):
        self.dataset.ignore_attributes = ["condition"]
        rval = self.dataset.get_data(include_ignore_attributes=True)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 39))
        rval, categorical = self.dataset.get_data(
            include_ignore_attributes=True, return_categorical_indicator=True)
        self.assertEqual(len(categorical), 39)
        rval = self.dataset.get_data(include_ignore_attributes=False)
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (898, 38))
        rval, categorical = self.dataset.get_data(
            include_ignore_attributes=False, return_categorical_indicator=True)
        self.assertEqual(len(categorical), 38)
        # TODO test multiple ignore attributes!

    def test_get_data_numpy(self):
        data = self.titanic.get_data(dataset_format='array')
        self.assertTrue(isinstance(data, np.ndarray))
        self.assertEqual(data.shape[1], len(self.titanic.features))
        self.assertEqual(data.shape[0], 1309)
        self.assertTrue(data.dtype == 'object')

        X, y = self.titanic.get_data(
            dataset_format='array',
            target=self.titanic.default_target_attribute)
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(X.shape, (1309, 13))
        self.assertEqual(y.shape, (1309,))
        self.assertTrue(X.dtype == 'object')
        self.assertTrue(y.dtype == int)

    def test_get_data_pandas(self):
        data = self.titanic.get_data(dataset_format='dataframe')
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertEqual(data.shape[1], len(self.titanic.features))
        self.assertEqual(data.shape[0], 1309)
        col_dtype = {
            'pclass': 'float64',
            'survived': 'category',
            'name': 'object',
            'sex': 'category',
            'age': 'float64',
            'sibsp': 'float64',
            'parch': 'float64',
            'ticket': 'object',
            'fare': 'float64',
            'cabin': 'object',
            'embarked': 'category',
            'boat': 'object',
            'body': 'float64',
            'home.dest': 'object'
        }
        for col_name in data.columns:
            print(data[col_name].dtype.name)
            print(col_dtype[col_name])
            self.assertTrue(data[col_name].dtype.name == col_dtype[col_name])

        X, y = self.titanic.get_data(
            dataset_format='dataframe',
            target=self.titanic.default_target_attribute)
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))
        self.assertEqual(X.shape, (1309, 13))
        self.assertEqual(y.shape, (1309,))
        for col_name in X.columns:
            self.assertTrue(X[col_name].dtype.name == col_dtype[col_name])
        self.assertTrue(y.dtype.name == col_dtype['survived'])

    def test_dataset_format_constructor(self):

        with catch_warnings():
            filterwarnings('error')
            self.assertRaises(
                DeprecationWarning,
                openml.OpenMLDataset,
                'Test',
                'Test',
                format='arff'
            )


class OpenMLDatasetTestOnTestServer(TestBase):
    def setUp(self):
        super(OpenMLDatasetTestOnTestServer, self).setUp()
        # longley, really small dataset
        self.dataset = openml.datasets.get_dataset(125)

    def test_tagging(self):
        tag = "testing_tag_{}_{}".format(self.id(), time())
        ds_list = openml.datasets.list_datasets(tag=tag)
        self.assertEqual(len(ds_list), 0)
        self.dataset.push_tag(tag)
        ds_list = openml.datasets.list_datasets(tag=tag)
        self.assertEqual(len(ds_list), 1)
        self.assertIn(125, ds_list)
        self.dataset.remove_tag(tag)
        ds_list = openml.datasets.list_datasets(tag=tag)
        self.assertEqual(len(ds_list), 0)


class OpenMLDatasetTestSparse(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super(OpenMLDatasetTestSparse, self).setUp()
        openml.config.server = self.production_server

        self.sparse_dataset = openml.datasets.get_dataset(4136)

    def test_get_sparse_dataset_with_target(self):
        X, y = self.sparse_dataset.get_data(target="class")
        self.assertTrue(sparse.issparse(X))
        self.assertEqual(X.dtype, np.float32)
        self.assertIsInstance(y, np.ndarray)
        self.assertIn(y.dtype, [np.int32, np.int64])
        self.assertEqual(X.shape, (600, 20000))
        X, y, attribute_names = self.sparse_dataset.get_data(
            target="class",
            return_attribute_names=True,
        )
        self.assertTrue(sparse.issparse(X))
        self.assertEqual(len(attribute_names), 20000)
        self.assertNotIn("class", attribute_names)
        self.assertEqual(y.shape, (600, ))

    def test_get_sparse_dataset(self):
        rval = self.sparse_dataset.get_data()
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual((600, 20001), rval.shape)
        rval, categorical = self.sparse_dataset.get_data(
            return_categorical_indicator=True)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(len(categorical), 20001)
        self.assertTrue(all([isinstance(cat, bool) for cat in categorical]))
        rval, attribute_names = self.sparse_dataset.get_data(
            return_attribute_names=True)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(len(attribute_names), 20001)
        self.assertTrue(all([isinstance(att, six.string_types)
                             for att in attribute_names]))

    def test_get_sparse_dataset_with_rowid(self):
        self.sparse_dataset.row_id_attribute = ["V256"]
        rval, categorical = self.sparse_dataset.get_data(
            include_row_id=True, return_categorical_indicator=True)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (600, 20001))
        self.assertEqual(len(categorical), 20001)
        rval, categorical = self.sparse_dataset.get_data(
            include_row_id=False, return_categorical_indicator=True)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (600, 20000))
        self.assertEqual(len(categorical), 20000)

    def test_get_sparse_dataset_with_ignore_attributes(self):
        self.sparse_dataset.ignore_attributes = ["V256"]
        rval = self.sparse_dataset.get_data(include_ignore_attributes=True)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (600, 20001))
        rval, categorical = self.sparse_dataset.get_data(
            include_ignore_attributes=True, return_categorical_indicator=True)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(len(categorical), 20001)
        rval = self.sparse_dataset.get_data(include_ignore_attributes=False)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(rval.dtype, np.float32)
        self.assertEqual(rval.shape, (600, 20000))
        rval, categorical = self.sparse_dataset.get_data(
            include_ignore_attributes=False, return_categorical_indicator=True)
        self.assertTrue(sparse.issparse(rval))
        self.assertEqual(len(categorical), 20000)
        # TODO test multiple ignore attributes!

    def test_get_sparse_dataset_rowid_and_ignore_and_target(self):
        # TODO: re-add row_id and ignore attributes
        self.sparse_dataset.ignore_attributes = ["V256"]
        self.sparse_dataset.row_id_attribute = ["V512"]
        X, y = self.sparse_dataset.get_data(
            target="class",
            include_row_id=False,
            include_ignore_attributes=False,
        )
        self.assertTrue(sparse.issparse(X))
        self.assertEqual(X.dtype, np.float32)
        self.assertIn(y.dtype, [np.int32, np.int64])
        self.assertEqual(X.shape, (600, 19998))
        X, y, categorical = self.sparse_dataset.get_data(
            target="class",
            return_categorical_indicator=True,
        )
        self.assertTrue(sparse.issparse(X))
        self.assertEqual(len(categorical), 19998)
        self.assertListEqual(categorical, [False] * 19998)
        self.assertEqual(y.shape, (600, ))


class OpenMLDatasetQualityTest(TestBase):
    def test__check_qualities(self):
        qualities = [{'oml:name': 'a', 'oml:value': '0.5'}]
        qualities = openml.datasets.dataset._check_qualities(qualities)
        self.assertEqual(qualities['a'], 0.5)

        qualities = [{'oml:name': 'a', 'oml:value': 'null'}]
        qualities = openml.datasets.dataset._check_qualities(qualities)
        self.assertNotEqual(qualities['a'], qualities['a'])

        qualities = [{'oml:name': 'a', 'oml:value': None}]
        qualities = openml.datasets.dataset._check_qualities(qualities)
        self.assertNotEqual(qualities['a'], qualities['a'])
