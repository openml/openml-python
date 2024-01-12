# License: BSD 3-Clause
from __future__ import annotations

import os
import unittest.mock
from time import time

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import openml
from openml.datasets import OpenMLDataFeature, OpenMLDataset
from openml.exceptions import PyOpenMLError
from openml.testing import TestBase


@pytest.mark.production()
class OpenMLDatasetTest(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()
        openml.config.server = self.production_server

        # Load dataset id 2 - dataset 2 is interesting because it contains
        # missing values, categorical features etc.
        self._dataset = None
        # titanic as missing values, categories, and string
        self._titanic = None
        # these datasets have some boolean features
        self._pc4 = None
        self._jm1 = None
        self._iris = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = openml.datasets.get_dataset(2, download_data=False)
        return self._dataset

    @property
    def titanic(self):
        if self._titanic is None:
            self._titanic = openml.datasets.get_dataset(40945, download_data=False)
        return self._titanic

    @property
    def pc4(self):
        if self._pc4 is None:
            self._pc4 = openml.datasets.get_dataset(1049, download_data=False)
        return self._pc4

    @property
    def jm1(self):
        if self._jm1 is None:
            self._jm1 = openml.datasets.get_dataset(1053, download_data=False)
        return self._jm1

    @property
    def iris(self):
        if self._iris is None:
            self._iris = openml.datasets.get_dataset(61, download_data=False)
        return self._iris

    def test_repr(self):
        # create a bare-bones dataset as would be returned by
        # create_dataset
        data = openml.datasets.OpenMLDataset(name="somename", description="a description")
        str(data)

    def test_init_string_validation(self):
        with pytest.raises(ValueError, match="Invalid symbols ' ' in name"):
            openml.datasets.OpenMLDataset(name="some name", description="a description")

        with pytest.raises(ValueError, match="Invalid symbols 'ï' in description"):
            openml.datasets.OpenMLDataset(name="somename", description="a descriptïon")

        with pytest.raises(ValueError, match="Invalid symbols 'ü' in citation"):
            openml.datasets.OpenMLDataset(
                name="somename",
                description="a description",
                citation="Something by Müller",
            )

    def test__unpack_categories_with_nan_likes(self):
        # unpack_categories decodes numeric categorical values according to the header
        # Containing a 'non' category in the header shouldn't lead to failure.
        categories = ["a", "b", None, float("nan"), np.nan]
        series = pd.Series([0, 1, None, float("nan"), np.nan, 1, 0])
        clean_series = OpenMLDataset._unpack_categories(series, categories)

        expected_values = ["a", "b", np.nan, np.nan, np.nan, "b", "a"]
        self.assertListEqual(list(clean_series.values), expected_values)
        self.assertListEqual(list(clean_series.cat.categories.values), list("ab"))

    def test_get_data_array(self):
        # Basic usage
        rval, _, categorical, attribute_names = self.dataset.get_data(dataset_format="array")
        assert isinstance(rval, np.ndarray)
        assert rval.dtype == np.float32
        assert rval.shape == (898, 39)
        assert len(categorical) == 39
        assert all(isinstance(cat, bool) for cat in categorical)
        assert len(attribute_names) == 39
        assert all(isinstance(att, str) for att in attribute_names)
        assert _ is None

        # check that an error is raised when the dataset contains string
        err_msg = "PyOpenML cannot handle string when returning numpy arrays"
        with pytest.raises(PyOpenMLError, match=err_msg):
            self.titanic.get_data(dataset_format="array")

    def test_get_data_pandas(self):
        data, _, _, _ = self.titanic.get_data(dataset_format="dataframe")
        assert isinstance(data, pd.DataFrame)
        assert data.shape[1] == len(self.titanic.features)
        assert data.shape[0] == 1309
        col_dtype = {
            "pclass": "uint8",
            "survived": "category",
            "name": "object",
            "sex": "category",
            "age": "float64",
            "sibsp": "uint8",
            "parch": "uint8",
            "ticket": "object",
            "fare": "float64",
            "cabin": "object",
            "embarked": "category",
            "boat": "object",
            "body": "float64",
            "home.dest": "object",
        }
        for col_name in data.columns:
            assert data[col_name].dtype.name == col_dtype[col_name]

        X, y, _, _ = self.titanic.get_data(
            dataset_format="dataframe",
            target=self.titanic.default_target_attribute,
        )
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (1309, 13)
        assert y.shape == (1309,)
        for col_name in X.columns:
            assert X[col_name].dtype.name == col_dtype[col_name]
        assert y.dtype.name == col_dtype["survived"]

    @pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
    def test_get_data_boolean_pandas(self):
        # test to check that we are converting properly True and False even
        # with some inconsistency when dumping the data on openml
        data, _, _, _ = self.jm1.get_data()
        assert data["defects"].dtype.name == "category"
        assert set(data["defects"].cat.categories) == {True, False}

        data, _, _, _ = self.pc4.get_data()
        assert data["c"].dtype.name == "category"
        assert set(data["c"].cat.categories) == {True, False}

    def test_get_data_no_str_data_for_nparrays(self):
        # check that an error is raised when the dataset contains string
        err_msg = "PyOpenML cannot handle string when returning numpy arrays"
        with pytest.raises(PyOpenMLError, match=err_msg):
            self.titanic.get_data(dataset_format="array")

    def _check_expected_type(self, dtype, is_cat, col):
        if is_cat:
            expected_type = "category"
        elif not col.isna().any() and (col.astype("uint8") == col).all():
            expected_type = "uint8"
        else:
            expected_type = "float64"

        assert dtype.name == expected_type

    @pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
    def test_get_data_with_rowid(self):
        self.dataset.row_id_attribute = "condition"
        rval, _, categorical, _ = self.dataset.get_data(include_row_id=True)
        assert isinstance(rval, pd.DataFrame)
        for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
            self._check_expected_type(dtype, is_cat, rval[col])
        assert rval.shape == (898, 39)
        assert len(categorical) == 39

        rval, _, categorical, _ = self.dataset.get_data()
        assert isinstance(rval, pd.DataFrame)
        for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
            self._check_expected_type(dtype, is_cat, rval[col])
        assert rval.shape == (898, 38)
        assert len(categorical) == 38

    def test_get_data_with_target_array(self):
        X, y, _, attribute_names = self.dataset.get_data(dataset_format="array", target="class")
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float32
        assert X.shape == (898, 38)
        assert y.dtype in [np.int32, np.int64]
        assert y.shape == (898,)
        assert len(attribute_names) == 38
        assert "class" not in attribute_names

    @pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
    def test_get_data_with_target_pandas(self):
        X, y, categorical, attribute_names = self.dataset.get_data(target="class")
        assert isinstance(X, pd.DataFrame)
        for dtype, is_cat, col in zip(X.dtypes, categorical, X):
            self._check_expected_type(dtype, is_cat, X[col])
        assert isinstance(y, pd.Series)
        assert y.dtype.name == "category"

        assert X.shape == (898, 38)
        assert len(attribute_names) == 38
        assert y.shape == (898,)

        assert "class" not in attribute_names

    def test_get_data_rowid_and_ignore_and_target(self):
        self.dataset.ignore_attribute = ["condition"]
        self.dataset.row_id_attribute = ["hardness"]
        X, y, categorical, names = self.dataset.get_data(target="class")
        assert X.shape == (898, 36)
        assert len(categorical) == 36
        cats = [True] * 3 + [False, True, True, False] + [True] * 23 + [False] * 3 + [True] * 3
        self.assertListEqual(categorical, cats)
        assert y.shape == (898,)

    @pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
    def test_get_data_with_ignore_attributes(self):
        self.dataset.ignore_attribute = ["condition"]
        rval, _, categorical, _ = self.dataset.get_data(include_ignore_attribute=True)
        for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
            self._check_expected_type(dtype, is_cat, rval[col])
        assert rval.shape == (898, 39)
        assert len(categorical) == 39

        rval, _, categorical, _ = self.dataset.get_data(include_ignore_attribute=False)
        for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
            self._check_expected_type(dtype, is_cat, rval[col])
        assert rval.shape == (898, 38)
        assert len(categorical) == 38

    def test_get_data_with_nonexisting_class(self):
        # This class is using the anneal dataset with labels [1, 2, 3, 4, 5, 'U']. However,
        # label 4 does not exist and we test that the features 5 and 'U' are correctly mapped to
        # indices 4 and 5, and that nothing is mapped to index 3.
        _, y, _, _ = self.dataset.get_data("class", dataset_format="dataframe")
        assert list(y.dtype.categories) == ["1", "2", "3", "4", "5", "U"]
        _, y, _, _ = self.dataset.get_data("class", dataset_format="array")
        assert np.min(y) == 0
        assert np.max(y) == 5
        # Check that no label is mapped to 3, since it is reserved for label '4'.
        assert np.sum(y == 3) == 0

    def test_get_data_corrupt_pickle(self):
        # Lazy loaded dataset, populate cache.
        self.iris.get_data()
        # Corrupt pickle file, overwrite as empty.
        with open(self.iris.data_pickle_file, "w") as fh:
            fh.write("")
        # Despite the corrupt file, the data should be loaded from the ARFF file.
        # A warning message is written to the python logger.
        xy, _, _, _ = self.iris.get_data()
        assert isinstance(xy, pd.DataFrame)
        assert xy.shape == (150, 5)

    def test_lazy_loading_metadata(self):
        # Initial Setup
        did_cache_dir = openml.utils._create_cache_directory_for_id(
            openml.datasets.functions.DATASETS_CACHE_DIR_NAME,
            2,
        )
        _compare_dataset = openml.datasets.get_dataset(
            2,
            download_data=False,
            download_features_meta_data=True,
            download_qualities=True,
        )
        change_time = os.stat(did_cache_dir).st_mtime

        # Test with cache
        _dataset = openml.datasets.get_dataset(
            2,
            download_data=False,
            download_features_meta_data=False,
            download_qualities=False,
        )
        assert change_time == os.stat(did_cache_dir).st_mtime
        assert _dataset.features == _compare_dataset.features
        assert _dataset.qualities == _compare_dataset.qualities

        # -- Test without cache
        openml.utils._remove_cache_dir_for_id(
            openml.datasets.functions.DATASETS_CACHE_DIR_NAME,
            did_cache_dir,
        )

        _dataset = openml.datasets.get_dataset(
            2,
            download_data=False,
            download_features_meta_data=False,
            download_qualities=False,
        )
        assert ["description.xml"] == os.listdir(did_cache_dir)
        assert change_time != os.stat(did_cache_dir).st_mtime
        assert _dataset.features == _compare_dataset.features
        assert _dataset.qualities == _compare_dataset.qualities


class OpenMLDatasetTestOnTestServer(TestBase):
    def setUp(self):
        super().setUp()
        # longley, really small dataset
        self.dataset = openml.datasets.get_dataset(125, download_data=False)

    def test_tagging(self):
        # tags can be at most 64 alphanumeric (+ underscore) chars
        unique_indicator = str(time()).replace(".", "")
        tag = f"test_tag_OpenMLDatasetTestOnTestServer_{unique_indicator}"
        datasets = openml.datasets.list_datasets(tag=tag, output_format="dataframe")
        assert datasets.empty
        self.dataset.push_tag(tag)
        datasets = openml.datasets.list_datasets(tag=tag, output_format="dataframe")
        assert len(datasets) == 1
        assert 125 in datasets["did"]
        self.dataset.remove_tag(tag)
        datasets = openml.datasets.list_datasets(tag=tag, output_format="dataframe")
        assert datasets.empty

    def test_get_feature_with_ontology_data_id_11(self):
        # test on car dataset, which has built-in ontology references
        dataset = openml.datasets.get_dataset(11)
        assert len(dataset.features) == 7
        assert len(dataset.features[1].ontologies) >= 2
        assert len(dataset.features[2].ontologies) >= 1
        assert len(dataset.features[3].ontologies) >= 1

    def test_add_remove_ontology_to_dataset(self):
        did = 1
        feature_index = 1
        ontology = 'https://www.openml.org/unittest/' + str(time())
        openml.datasets.functions.data_feature_add_ontology(did, feature_index, ontology)
        openml.datasets.functions.data_feature_remove_ontology(did, feature_index, ontology)

    def test_add_same_ontology_multiple_features(self):
        did = 1
        ontology = 'https://www.openml.org/unittest/' + str(time())

        for i in range(3):
            openml.datasets.functions.data_feature_add_ontology(did, i, ontology)


    def test_add_illegal_long_ontology(self):
        did = 1
        ontology = 'http://www.google.com/' + ('a' * 257)
        try:
            openml.datasets.functions.data_feature_add_ontology(did, 1, ontology)
            assert False
        except openml.exceptions.OpenMLServerException as e:
            assert e.code == 1105

    def test_add_illegal_url_ontology(self):
        did = 1
        ontology = 'not_a_url' + str(time())
        try:
            openml.datasets.functions.data_feature_add_ontology(did, 1, ontology)
            assert False
        except openml.exceptions.OpenMLServerException as e:
            assert e.code == 1106

@pytest.mark.production()
class OpenMLDatasetTestSparse(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()
        openml.config.server = self.production_server

        self.sparse_dataset = openml.datasets.get_dataset(4136, download_data=False)

    def test_get_sparse_dataset_array_with_target(self):
        X, y, _, attribute_names = self.sparse_dataset.get_data(
            dataset_format="array",
            target="class",
        )

        assert sparse.issparse(X)
        assert X.dtype == np.float32
        assert X.shape == (600, 20000)

        assert isinstance(y, np.ndarray)
        assert y.dtype in [np.int32, np.int64]
        assert y.shape == (600,)

        assert len(attribute_names) == 20000
        assert "class" not in attribute_names

    def test_get_sparse_dataset_dataframe_with_target(self):
        X, y, _, attribute_names = self.sparse_dataset.get_data(
            dataset_format="dataframe",
            target="class",
        )
        assert isinstance(X, pd.DataFrame)
        assert isinstance(X.dtypes[0], pd.SparseDtype)
        assert X.shape == (600, 20000)

        assert isinstance(y, pd.Series)
        assert isinstance(y.dtypes, pd.SparseDtype)
        assert y.shape == (600,)

        assert len(attribute_names) == 20000
        assert "class" not in attribute_names

    def test_get_sparse_dataset_array(self):
        rval, _, categorical, attribute_names = self.sparse_dataset.get_data(dataset_format="array")
        assert sparse.issparse(rval)
        assert rval.dtype == np.float32
        assert rval.shape == (600, 20001)

        assert len(categorical) == 20001
        assert all(isinstance(cat, bool) for cat in categorical)

        assert len(attribute_names) == 20001
        assert all(isinstance(att, str) for att in attribute_names)

    def test_get_sparse_dataset_dataframe(self):
        rval, *_ = self.sparse_dataset.get_data()
        assert isinstance(rval, pd.DataFrame)
        np.testing.assert_array_equal(
            [pd.SparseDtype(np.float32, fill_value=0.0)] * len(rval.dtypes),
            rval.dtypes,
        )
        assert rval.shape == (600, 20001)

    def test_get_sparse_dataset_with_rowid(self):
        self.sparse_dataset.row_id_attribute = ["V256"]
        rval, _, categorical, _ = self.sparse_dataset.get_data(
            dataset_format="array",
            include_row_id=True,
        )
        assert sparse.issparse(rval)
        assert rval.dtype == np.float32
        assert rval.shape == (600, 20001)
        assert len(categorical) == 20001

        rval, _, categorical, _ = self.sparse_dataset.get_data(
            dataset_format="array",
            include_row_id=False,
        )
        assert sparse.issparse(rval)
        assert rval.dtype == np.float32
        assert rval.shape == (600, 20000)
        assert len(categorical) == 20000

    def test_get_sparse_dataset_with_ignore_attributes(self):
        self.sparse_dataset.ignore_attribute = ["V256"]
        rval, _, categorical, _ = self.sparse_dataset.get_data(
            dataset_format="array",
            include_ignore_attribute=True,
        )
        assert sparse.issparse(rval)
        assert rval.dtype == np.float32
        assert rval.shape == (600, 20001)

        assert len(categorical) == 20001
        rval, _, categorical, _ = self.sparse_dataset.get_data(
            dataset_format="array",
            include_ignore_attribute=False,
        )
        assert sparse.issparse(rval)
        assert rval.dtype == np.float32
        assert rval.shape == (600, 20000)
        assert len(categorical) == 20000

    def test_get_sparse_dataset_rowid_and_ignore_and_target(self):
        # TODO: re-add row_id and ignore attributes
        self.sparse_dataset.ignore_attribute = ["V256"]
        self.sparse_dataset.row_id_attribute = ["V512"]
        X, y, categorical, _ = self.sparse_dataset.get_data(
            dataset_format="array",
            target="class",
            include_row_id=False,
            include_ignore_attribute=False,
        )
        assert sparse.issparse(X)
        assert X.dtype == np.float32
        assert y.dtype in [np.int32, np.int64]
        assert X.shape == (600, 19998)

        assert len(categorical) == 19998
        self.assertListEqual(categorical, [False] * 19998)
        assert y.shape == (600,)

    def test_get_sparse_categorical_data_id_395(self):
        dataset = openml.datasets.get_dataset(395, download_data=True)
        feature = dataset.features[3758]
        assert isinstance(dataset, OpenMLDataset)
        assert isinstance(feature, OpenMLDataFeature)
        assert dataset.name == "re1.wc"
        assert feature.name == "CLASS_LABEL"
        assert feature.data_type == "nominal"
        assert len(feature.nominal_values) == 25


class OpenMLDatasetFunctionTest(TestBase):
    @unittest.mock.patch("openml.datasets.dataset.pickle")
    @unittest.mock.patch("openml.datasets.dataset._get_features_pickle_file")
    def test__read_features(self, filename_mock, pickle_mock):
        """Test we read the features from the xml if no cache pickle is available.

        This test also does some simple checks to verify that the features are read correctly
        """
        filename_mock.return_value = os.path.join(self.workdir, "features.xml.pkl")
        pickle_mock.load.side_effect = FileNotFoundError
        features = openml.datasets.dataset._read_features(
            os.path.join(
                self.static_cache_dir,
                "org",
                "openml",
                "test",
                "datasets",
                "2",
                "features.xml",
            ),
        )
        assert isinstance(features, dict)
        assert len(features) == 39
        assert isinstance(features[0], OpenMLDataFeature)
        assert features[0].name == "family"
        assert len(features[0].nominal_values) == 9
        # pickle.load is never called because the features pickle file didn't exist
        assert pickle_mock.load.call_count == 0
        assert pickle_mock.dump.call_count == 1

    @unittest.mock.patch("openml.datasets.dataset.pickle")
    @unittest.mock.patch("openml.datasets.dataset._get_qualities_pickle_file")
    def test__read_qualities(self, filename_mock, pickle_mock):
        """Test we read the qualities from the xml if no cache pickle is available.

        This test also does some minor checks to ensure that the qualities are read correctly.
        """
        filename_mock.return_value = os.path.join(self.workdir, "qualities.xml.pkl")
        pickle_mock.load.side_effect = FileNotFoundError
        qualities = openml.datasets.dataset._read_qualities(
            os.path.join(
                self.static_cache_dir,
                "org",
                "openml",
                "test",
                "datasets",
                "2",
                "qualities.xml",
            ),
        )
        assert isinstance(qualities, dict)
        assert len(qualities) == 106
        # pickle.load is never called because the qualities pickle file didn't exist
        assert pickle_mock.load.call_count == 0
        assert pickle_mock.dump.call_count == 1

    def test__check_qualities(self):
        qualities = [{"oml:name": "a", "oml:value": "0.5"}]
        qualities = openml.datasets.dataset._check_qualities(qualities)
        assert qualities["a"] == 0.5

        qualities = [{"oml:name": "a", "oml:value": "null"}]
        qualities = openml.datasets.dataset._check_qualities(qualities)
        assert qualities["a"] != qualities["a"]

        qualities = [{"oml:name": "a", "oml:value": None}]
        qualities = openml.datasets.dataset._check_qualities(qualities)
        assert qualities["a"] != qualities["a"]
