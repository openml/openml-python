# License: BSD 3-Clause
from __future__ import annotations

import os
from time import time

import numpy as np
import pandas as pd
import pytest

import openml
from openml.datasets import OpenMLDataFeature, OpenMLDataset
from openml.exceptions import PyOpenMLError


# Fixtures for datasets
@pytest.fixture(scope="module")
def production_server():
    """Configure to use production server."""
    original_server = openml.config.server
    openml.config.server = "https://www.openml.org/api/v1/xml"
    yield
    openml.config.server = original_server


@pytest.fixture(scope="module")
def anneal_dataset(production_server):
    """Load dataset id 2 - contains missing values, categorical features etc."""
    return openml.datasets.get_dataset(2, download_data=False)


@pytest.fixture(scope="module")
def titanic_dataset(production_server):
    """Titanic dataset - has missing values, categories, and string."""
    return openml.datasets.get_dataset(40945, download_data=False)


@pytest.fixture(scope="module")
def pc4_dataset(production_server):
    """PC4 dataset - has some boolean features."""
    return openml.datasets.get_dataset(1049, download_data=False)


@pytest.fixture(scope="module")
def jm1_dataset(production_server):
    """JM1 dataset - has some boolean features."""
    return openml.datasets.get_dataset(1053, download_data=False)


@pytest.fixture(scope="module")
def iris_dataset(production_server):
    """Iris dataset."""
    return openml.datasets.get_dataset(61, download_data=False)


def test_repr():
    """Test dataset repr doesn't crash."""
    # create a bare-bones dataset as would be returned by
    # create_dataset
    data = openml.datasets.OpenMLDataset(name="somename", description="a description")
    str(data)


def test_init_string_validation():
    """Test that invalid characters in dataset fields raise ValueError."""
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


def test__unpack_categories_with_nan_likes():
    """Test that unpacking categories with NaN-like values works correctly."""
    # unpack_categories decodes numeric categorical values according to the header
    # Containing a 'non' category in the header shouldn't lead to failure.
    categories = ["a", "b", None, float("nan"), np.nan]
    series = pd.Series([0, 1, None, float("nan"), np.nan, 1, 0])
    clean_series = OpenMLDataset._unpack_categories(series, categories)

    expected_values = ["a", "b", np.nan, np.nan, np.nan, "b", "a"]
    assert list(clean_series.values) == expected_values
    assert list(clean_series.cat.categories.values) == list("ab")

@pytest.mark.production()
def test_get_data_pandas(titanic_dataset):
    """Test get_data returns correct pandas DataFrame."""
    data, _, _, _ = titanic_dataset.get_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] == len(titanic_dataset.features)
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

    X, y, _, _ = titanic_dataset.get_data(
        target=titanic_dataset.default_target_attribute,
    )
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (1309, 13)
    assert y.shape == (1309,)
    for col_name in X.columns:
        assert X[col_name].dtype.name == col_dtype[col_name]
    assert y.dtype.name == col_dtype["survived"]


@pytest.mark.production()
@pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
def test_get_data_boolean_pandas(jm1_dataset, pc4_dataset):
    """Test boolean data is correctly converted."""
    # test to check that we are converting properly True and False even
    # with some inconsistency when dumping the data on openml
    data, _, _, _ = jm1_dataset.get_data()
    assert data["defects"].dtype.name == "category"
    assert set(data["defects"].cat.categories) == {True, False}

    data, _, _, _ = pc4_dataset.get_data()
    assert data["c"].dtype.name == "category"
    assert set(data["c"].cat.categories) == {True, False}


def _check_expected_type(dtype, is_cat, col):
    """Helper to check expected pandas dtype."""
    if is_cat:
        expected_type = "category"
    elif not col.isna().any() and (col.astype("uint8") == col).all():
        expected_type = "uint8"
    else:
        expected_type = "float64"

    assert dtype.name == expected_type

@pytest.mark.production()
@pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
def test_get_data_with_rowid(anneal_dataset):
    """Test get_data with row_id_attribute."""
    anneal_dataset.row_id_attribute = "condition"
    rval, _, categorical, _ = anneal_dataset.get_data(include_row_id=True)
    assert isinstance(rval, pd.DataFrame)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 39)
    assert len(categorical) == 39

    rval, _, categorical, _ = anneal_dataset.get_data()
    assert isinstance(rval, pd.DataFrame)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 38)
    assert len(categorical) == 38


@pytest.mark.production()
@pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
def test_get_data_with_target_pandas(anneal_dataset):
    """Test get_data with target specified."""
    X, y, categorical, attribute_names = anneal_dataset.get_data(target="class")
    assert isinstance(X, pd.DataFrame)
    for dtype, is_cat, col in zip(X.dtypes, categorical, X):
        _check_expected_type(dtype, is_cat, X[col])
    assert isinstance(y, pd.Series)
    assert y.dtype.name == "category"

    assert X.shape == (898, 38)
    assert len(attribute_names) == 38
    assert y.shape == (898,)

    assert "class" not in attribute_names


@pytest.mark.production()
def test_get_data_rowid_and_ignore_and_target(anneal_dataset):
    """Test get_data with rowid, ignore and target attributes."""
    anneal_dataset.ignore_attribute = ["condition"]
    anneal_dataset.row_id_attribute = ["hardness"]
    X, y, categorical, names = anneal_dataset.get_data(target="class")
    assert X.shape == (898, 36)
    assert len(categorical) == 36
    cats = [True] * 3 + [False, True, True, False] + [True] * 23 + [False] * 3 + [True] * 3
    assert categorical == cats
    assert y.shape == (898,)


@pytest.mark.production()
@pytest.mark.skip("https://github.com/openml/openml-python/issues/1157")
def test_get_data_with_ignore_attributes(anneal_dataset):
    """Test get_data with ignore_attribute."""
    anneal_dataset.ignore_attribute = ["condition"]
    rval, _, categorical, _ = anneal_dataset.get_data(include_ignore_attribute=True)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 39)
    assert len(categorical) == 39

    rval, _, categorical, _ = anneal_dataset.get_data(include_ignore_attribute=False)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 38)
    assert len(categorical) == 38


@pytest.mark.production()
def test_get_data_with_nonexisting_class(anneal_dataset):
    """Test dataset with missing class labels."""
    # This class is using the anneal dataset with labels [1, 2, 3, 4, 5, 'U']. However,
    # label 4 does not exist and we test that the features 5 and 'U' are correctly mapped to
    # indices 4 and 5, and that nothing is mapped to index 3.
    _, y, _, _ = anneal_dataset.get_data("class")
    assert list(y.dtype.categories) == ["1", "2", "3", "4", "5", "U"]


@pytest.mark.production()
def test_get_data_corrupt_pickle(iris_dataset):
    """Test that corrupt pickle files are handled gracefully."""
    # Lazy loaded dataset, populate cache.
    iris_dataset.get_data()
    # Corrupt pickle file, overwrite as empty.
    with open(iris_dataset.data_pickle_file, "w") as fh:
        fh.write("")
    # Despite the corrupt file, the data should be loaded from the ARFF file.
    # A warning message is written to the python logger.
    xy, _, _, _ = iris_dataset.get_data()
    assert isinstance(xy, pd.DataFrame)
    assert xy.shape == (150, 5)


@pytest.mark.production()
def test_lazy_loading_metadata():
    """Test lazy loading of dataset metadata."""
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


@pytest.mark.production()
def test_equality_comparison(iris_dataset, titanic_dataset):
    """Test dataset equality comparison."""
    assert iris_dataset == iris_dataset
    assert iris_dataset != titanic_dataset
    assert titanic_dataset != "Wrong_object"


def test_tagging():
    dataset = openml.datasets.get_dataset(125, download_data=False)

    # tags can be at most 64 alphanumeric (+ underscore) chars
    unique_indicator = str(time()).replace(".", "")
    tag = f"test_tag_OpenMLDatasetTestOnTestServer_{unique_indicator}"
    datasets = openml.datasets.list_datasets(tag=tag)
    assert datasets.empty
    dataset.push_tag(tag)
    datasets = openml.datasets.list_datasets(tag=tag)
    assert len(datasets) == 1
    assert 125 in datasets["did"]
    dataset.remove_tag(tag)
    datasets = openml.datasets.list_datasets(tag=tag)
    assert datasets.empty

def test_get_feature_with_ontology_data_id_11():
    # test on car dataset, which has built-in ontology references
    dataset = openml.datasets.get_dataset(11)
    assert len(dataset.features) == 7
    assert len(dataset.features[1].ontologies) >= 2
    assert len(dataset.features[2].ontologies) >= 1
    assert len(dataset.features[3].ontologies) >= 1   

def test_add_remove_ontology_to_dataset():
    did = 1
    feature_index = 1
    ontology = "https://www.openml.org/unittest/" + str(time())
    openml.datasets.functions.data_feature_add_ontology(did, feature_index, ontology)
    openml.datasets.functions.data_feature_remove_ontology(did, feature_index, ontology)    

def test_add_same_ontology_multiple_features():
    did = 1
    ontology = "https://www.openml.org/unittest/" + str(time())

    for i in range(3):
        openml.datasets.functions.data_feature_add_ontology(did, i, ontology)    


def test_add_illegal_long_ontology():
    did = 1
    ontology = "http://www.google.com/" + ("a" * 257)
    try:
        openml.datasets.functions.data_feature_add_ontology(did, 1, ontology)
        assert False
    except openml.exceptions.OpenMLServerException as e:
        assert e.code == 1105
    


def test_add_illegal_url_ontology():
    did = 1
    ontology = "not_a_url" + str(time())
    try:
        openml.datasets.functions.data_feature_add_ontology(did, 1, ontology)
        assert False
    except openml.exceptions.OpenMLServerException as e:
        assert e.code == 1106


@pytest.fixture(scope="module")
def sparse_dataset(production_server):
    """Sparse dataset for testing."""
    return openml.datasets.get_dataset(4136, download_data=False)


@pytest.mark.production()
def test_get_sparse_dataset_dataframe_with_target(sparse_dataset):
    """Test sparse dataset with target."""
    X, y, _, attribute_names = sparse_dataset.get_data(target="class")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(X.dtypes[0], pd.SparseDtype)
    assert X.shape == (600, 20000)

    assert isinstance(y, pd.Series)
    assert isinstance(y.dtypes, pd.SparseDtype)
    assert y.shape == (600,)

    assert len(attribute_names) == 20000
    assert "class" not in attribute_names


@pytest.mark.production()
def test_get_sparse_dataset_dataframe(sparse_dataset):
    """Test sparse dataset returns sparse DataFrame."""
    rval, *_ = sparse_dataset.get_data()
    assert isinstance(rval, pd.DataFrame)
    np.testing.assert_array_equal(
        [pd.SparseDtype(np.float32, fill_value=0.0)] * len(rval.dtypes),
        rval.dtypes,
    )
    assert rval.shape == (600, 20001)


@pytest.mark.production()
def test_get_sparse_dataset_rowid_and_ignore_and_target(sparse_dataset):
    """Test sparse dataset with row_id, ignore and target attributes."""
    # TODO: re-add row_id and ignore attributes
    sparse_dataset.ignore_attribute = ["V256"]
    sparse_dataset.row_id_attribute = ["V512"]
    X, y, categorical, _ = sparse_dataset.get_data(
        target="class",
        include_row_id=False,
        include_ignore_attribute=False,
    )
    assert all(dtype == pd.SparseDtype(np.float32, fill_value=0.0) for dtype in X.dtypes)
    # array format returned dense, but now we only return sparse and let the user handle it.
    assert isinstance(y.dtypes, pd.SparseDtype)
    assert X.shape == (600, 19998)

    assert len(categorical) == 19998
    assert categorical == [False] * 19998
    assert y.shape == (600,)


@pytest.mark.production()
def test_get_sparse_categorical_data_id_395():
    """Test sparse categorical dataset."""
    dataset = openml.datasets.get_dataset(395, download_data=True)
    feature = dataset.features[3758]
    assert isinstance(dataset, OpenMLDataset)
    assert isinstance(feature, OpenMLDataFeature)
    assert dataset.name == "re1.wc"
    assert feature.name == "CLASS_LABEL"
    assert feature.data_type == "nominal"
    assert len(feature.nominal_values) == 25


def test__read_features(mocker, workdir, static_cache_dir):
    """Test we read the features from the xml if no cache pickle is available.
    This test also does some simple checks to verify that the features are read correctly
    """
    filename_mock = mocker.patch("openml.datasets.dataset._get_features_pickle_file")
    pickle_mock = mocker.patch("openml.datasets.dataset.pickle")

    filename_mock.return_value = os.path.join(workdir, "features.xml.pkl")
    pickle_mock.load.side_effect = FileNotFoundError

    features = openml.datasets.dataset._read_features(
        os.path.join(
            static_cache_dir,
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


def test__read_qualities(static_cache_dir, workdir, mocker):
    """Test we read the qualities from the xml if no cache pickle is available.
    This test also does some minor checks to ensure that the qualities are read correctly.
    """

    filename_mock = mocker.patch("openml.datasets.dataset._get_qualities_pickle_file")
    pickle_mock = mocker.patch("openml.datasets.dataset.pickle")

    filename_mock.return_value=os.path.join(workdir, "qualities.xml.pkl")
    pickle_mock.load.side_effect = FileNotFoundError

    qualities = openml.datasets.dataset._read_qualities(
        os.path.join(
            static_cache_dir,
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
    assert pickle_mock.load.call_count == 0
    assert pickle_mock.dump.call_count == 1



def test__check_qualities():
    qualities = [{"oml:name": "a", "oml:value": "0.5"}]
    qualities = openml.datasets.dataset._check_qualities(qualities)
    assert qualities["a"] == 0.5

    qualities = [{"oml:name": "a", "oml:value": "null"}]
    qualities = openml.datasets.dataset._check_qualities(qualities)
    assert qualities["a"] != qualities["a"]

    qualities = [{"oml:name": "a", "oml:value": None}]
    qualities = openml.datasets.dataset._check_qualities(qualities)
    assert qualities["a"] != qualities["a"]