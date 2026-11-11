# License: BSD 3-Clause
from __future__ import annotations

import os
import shutil
from time import time

import numpy as np
import pandas as pd
import pytest
import requests

import openml
from openml.datasets import OpenMLDataFeature, OpenMLDataset

SPARSE_DATASET_ID = 4136


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


@pytest.mark.production_server
def test_get_data_pandas():
    """Test get_data returns correct pandas DataFrame."""
    titanic_dataset = openml.datasets.get_dataset(40945)
    data, _, _, _ = titanic_dataset.get_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] == len(titanic_dataset.features)
    assert data.shape[0] == 1309
    # Dynamically detect what this version of Pandas calls string columns.
    str_dtype = data["name"].dtype.name

    col_dtype = {
        "pclass": "uint8",
        "survived": "category",
        "name": str_dtype,
        "sex": "category",
        "age": "float64",
        "sibsp": "uint8",
        "parch": "uint8",
        "ticket": str_dtype,
        "fare": "float64",
        "cabin": str_dtype,
        "embarked": "category",
        "boat": str_dtype,
        "body": "float64",
        "home.dest": str_dtype,
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


@pytest.mark.production_server
def test_get_data_boolean_pandas(monkeypatch):
    """Test boolean data is converted to categorical when loading from arff."""
    # At some point we decided to convert bool-ey features to
    # categories, because from the ARFF file we technically can't
    # specify which is which (#1106, #1157).
    # This seems mostly historical after conversion to parquet,
    # which does load as boolean.
    monkeypatch.setenv(openml.config.OPENML_SKIP_PARQUET_ENV_VAR, "true")
    jm1_dataset = openml.datasets.get_dataset(1053)
    data, _, _, _ = jm1_dataset.get_data()
    assert data["defects"].dtype.name == "category"
    assert set(data["defects"].cat.categories) == {True, False}

    pc4_dataset = openml.datasets.get_dataset(1049)
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


def test_get_data_with_rowid(static_cache_dir):
    """Test get_data with row_id_attribute."""
    openml.config.set_root_cache_directory(static_cache_dir)
    anneal_dataset = openml.datasets.get_dataset(2)

    anneal_dataset.row_id_attribute = "condition"
    rval, _, categorical, _ = anneal_dataset.get_data(include_row_id=True)
    assert isinstance(rval, pd.DataFrame)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval, strict=False):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 39)
    assert len(categorical) == 39

    rval, _, categorical, _ = anneal_dataset.get_data()
    assert isinstance(rval, pd.DataFrame)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval, strict=False):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 38)
    assert len(categorical) == 38


def test_get_data_with_target_pandas(static_cache_dir):
    """Test get_data with target specified."""
    openml.config.set_root_cache_directory(static_cache_dir)
    anneal_dataset = openml.datasets.get_dataset(2)
    X, y, categorical, attribute_names = anneal_dataset.get_data(target="class")
    assert isinstance(X, pd.DataFrame)
    for dtype, is_cat, col in zip(X.dtypes, categorical, X, strict=False):
        _check_expected_type(dtype, is_cat, X[col])
    assert isinstance(y, pd.Series)
    assert y.dtype.name == "category"

    assert X.shape == (898, 38)
    assert len(attribute_names) == 38
    assert y.shape == (898,)

    assert "class" not in attribute_names


def test_get_data_rowid_and_ignore_and_target(static_cache_dir):
    """Test get_data with rowid, ignore and target attributes."""
    openml.config.set_root_cache_directory(static_cache_dir)
    anneal_dataset = openml.datasets.get_dataset(2)
    anneal_dataset.ignore_attribute = ["condition"]
    anneal_dataset.row_id_attribute = ["hardness"]
    X, y, categorical, _names = anneal_dataset.get_data(target="class")
    assert X.shape == (898, 36)
    assert len(categorical) == 36
    cats = [True] * 3 + [False, True, True, False] + [True] * 23 + [False] * 3 + [True] * 3
    assert categorical == cats
    assert y.shape == (898,)


def test_get_data_with_ignore_attributes(static_cache_dir):
    """Test get_data with ignore_attribute."""
    openml.config.set_root_cache_directory(static_cache_dir)
    anneal_dataset = openml.datasets.get_dataset(2)
    anneal_dataset.ignore_attribute = ["condition"]
    rval, _, categorical, _ = anneal_dataset.get_data(include_ignore_attribute=True)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval, strict=False):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 39)
    assert len(categorical) == 39

    rval, _, categorical, _ = anneal_dataset.get_data(include_ignore_attribute=False)
    for dtype, is_cat, col in zip(rval.dtypes, categorical, rval, strict=False):
        _check_expected_type(dtype, is_cat, rval[col])
    assert rval.shape == (898, 38)
    assert len(categorical) == 38


@pytest.mark.test_server
def test_get_data_corrupt_pickle():
    """Test that corrupt pickle files are handled gracefully.

    The pickle file is stored for fast-load convenience only, but the
    primary file is the parquet (or aff) file. If a pickle file is corrupted
    the file should be loaded from the parquet (or arff) file instead.

    This feature will be removed when arff support is removed (#1027).
    """
    iris_dataset = openml.datasets.get_dataset(128, download_data=True)
    # Only on first load would the pickle file actually be created.
    iris_dataset.get_data()
    assert os.stat(iris_dataset.data_pickle_file).st_size > 0

    with open(iris_dataset.data_pickle_file, "w") as fh:
        fh.write("")
    assert os.stat(iris_dataset.data_pickle_file).st_size == 0

    # Despite the corrupt file, the data should be loaded from the ARFF file.
    # A warning message is written to the python logger.
    xy, _, _, _ = iris_dataset.get_data()
    assert isinstance(xy, pd.DataFrame)
    assert xy.shape == (150, 5)


@pytest.mark.test_server
def test_lazy_loading_metadata(mocker):
    """Test lazy loading of dataset metadata."""
    # The test automatically executes with a temporary cache directory,
    # that means files should be downloaded:
    spy = mocker.spy(requests.sessions.Session, "request")
    _compare_dataset = openml.datasets.get_dataset(
        2,
        download_data=False,
        download_features_meta_data=True,
        download_qualities=True,
    )
    eager_download_call_count = spy.call_count
    assert eager_download_call_count > 0, (
        "Eager download should make network call if cache is empty."
    )

    _dataset = openml.datasets.get_dataset(
        2,
        download_data=False,
        download_features_meta_data=False,
        download_qualities=False,
    )

    assert spy.call_count == eager_download_call_count, (
        "Lazy instantiation with description in cache shouldn't make network call."
    )

    assert _dataset.features == _compare_dataset.features
    assert _dataset.qualities == pytest.approx(_compare_dataset.qualities, nan_ok=True)
    assert spy.call_count == eager_download_call_count, (
        "Lazy loading with files in cache shouldn't make network calls."
    )

    # -- Test without cache
    shutil.rmtree(openml.config.get_cache_directory())
    _dataset = openml.datasets.get_dataset(
        2,
        download_data=False,
        download_features_meta_data=False,
        download_qualities=False,
    )

    assert _dataset.features == _compare_dataset.features
    assert _dataset.qualities == pytest.approx(_compare_dataset.qualities, nan_ok=True)
    lazy_download_call_count = spy.call_count - eager_download_call_count
    assert lazy_download_call_count == eager_download_call_count, (
        "Lazy fetching of files should result in the same number of calls "
        "if all attributes have been accessed."
    )


@pytest.mark.test_server
def test_equality_comparison():
    """Test dataset equality comparison."""
    dataset_a = openml.datasets.get_dataset(1)
    dataset_b = openml.datasets.get_dataset(2)

    assert dataset_a == dataset_a
    assert dataset_a != dataset_b
    assert dataset_a != "not any dataset"


@pytest.mark.test_server
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


@pytest.mark.test_server
def test_get_feature_with_ontology_data_id_11():
    # test on car dataset, which has built-in ontology references
    dataset = openml.datasets.get_dataset(11)
    assert len(dataset.features) == 7
    assert len(dataset.features[1].ontologies) >= 2
    assert len(dataset.features[2].ontologies) >= 1
    assert len(dataset.features[3].ontologies) >= 1


@pytest.mark.test_server
def test_add_remove_ontology_to_dataset():
    did = 1
    feature_index = 1
    ontology = "https://www.openml.org/unittest/" + str(time())
    openml.datasets.functions.data_feature_add_ontology(did, feature_index, ontology)
    openml.datasets.functions.data_feature_remove_ontology(did, feature_index, ontology)


@pytest.mark.test_server
def test_add_same_ontology_multiple_features():
    did = 1
    ontology = "https://www.openml.org/unittest/" + str(time())

    for i in range(3):
        openml.datasets.functions.data_feature_add_ontology(did, i, ontology)


@pytest.mark.test_server
def test_add_illegal_long_ontology():
    did = 1
    ontology = "http://www.google.com/" + ("a" * 257)
    try:
        openml.datasets.functions.data_feature_add_ontology(did, 1, ontology)
        raise AssertionError()
    except openml.exceptions.OpenMLServerException as e:
        assert e.code == 1105  # noqa: PT017


@pytest.mark.test_server
def test_add_illegal_url_ontology():
    did = 1
    ontology = "not_a_url" + str(time())
    try:
        openml.datasets.functions.data_feature_add_ontology(did, 1, ontology)
        raise AssertionError()
    except openml.exceptions.OpenMLServerException as e:
        assert e.code == 1106  # noqa: PT017


@pytest.mark.production_server
def test_get_sparse_dataset_dataframe_with_target():
    """Test sparse dataset with target."""
    sparse_dataset = openml.datasets.get_dataset(4136, download_data=False)
    X, y, _, attribute_names = sparse_dataset.get_data(target="class")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(X.dtypes.iloc[0], pd.SparseDtype)
    assert X.shape == (600, 20000)

    assert isinstance(y, pd.Series)
    assert isinstance(y.dtypes, pd.SparseDtype)
    assert y.shape == (600,)

    assert len(attribute_names) == 20000
    assert "class" not in attribute_names


@pytest.mark.production_server
def test_get_sparse_dataset_dataframe():
    sparse_dataset = openml.datasets.get_dataset(4136, download_data=False)
    """Test sparse dataset returns sparse DataFrame."""
    rval, *_ = sparse_dataset.get_data()
    assert isinstance(rval, pd.DataFrame)
    np.testing.assert_array_equal(
        [pd.SparseDtype(np.float32, fill_value=0.0)] * len(rval.dtypes),
        rval.dtypes,
    )
    assert rval.shape == (600, 20001)


@pytest.mark.production_server
def test_get_sparse_dataset_rowid_and_ignore_and_target():
    """Test sparse dataset with row_id, ignore and target attributes."""
    sparse_dataset = openml.datasets.get_dataset(4136, download_data=False)
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


@pytest.mark.production_server
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


@pytest.mark.test_server
def test__read_features(mocker, workdir, static_cache_dir):
    """Test we read the features from the xml if no cache pickle is available.
    This test also does some simple checks to verify that the features are read correctly
    """
    filename_mock = mocker.patch("openml.datasets.dataset._get_features_pickle_file")
    pickle_mock = mocker.patch("openml._api.resources.dataset.pickle")

    filename_mock.return_value = os.path.join(workdir, "features.xml.pkl")  # noqa: PTH118
    pickle_mock.load.side_effect = FileNotFoundError

    features = openml.datasets.dataset._read_features(
        os.path.join(  # noqa: PTH118
            static_cache_dir,
            "org",
            "openml",
            "test",
            "api",
            "v1",
            "xml",
            "data",
            "features",
            "2",
            "body.xml",
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


@pytest.mark.test_server
def test__read_qualities(static_cache_dir, workdir, mocker):
    """Test we read the qualities from the xml if no cache pickle is available.
    This test also does some minor checks to ensure that the qualities are read correctly.
    """
    filename_mock = mocker.patch("openml.datasets.dataset._get_qualities_pickle_file")
    pickle_mock = mocker.patch("openml._api.resources.dataset.pickle")

    filename_mock.return_value = os.path.join(workdir, "qualities.xml.pkl")  # noqa: PTH118
    pickle_mock.load.side_effect = FileNotFoundError

    qualities = openml.datasets.dataset._read_qualities(
        os.path.join(  # noqa: PTH118
            static_cache_dir,
            "org",
            "openml",
            "test",
            "api",
            "v1",
            "xml",
            "data",
            "qualities",
            "2",
            "body.xml",
        ),
    )
    assert isinstance(qualities, dict)
    assert len(qualities) == 106
    assert pickle_mock.load.call_count == 0
    assert pickle_mock.dump.call_count == 1
