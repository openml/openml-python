# License: BSD 3-Clause
from __future__ import annotations

import os
from pathlib import Path
import random
import shutil
import time
from itertools import product
from unittest import mock

import arff
import numpy as np
import pandas as pd
import pytest
import requests
import scipy.sparse
from oslo_concurrency import lockutils

import openml
from openml import OpenMLDataset
from openml._api_calls import _download_minio_file
from openml.datasets import edit_dataset, fork_dataset
from openml.datasets.functions import (
    DATASETS_CACHE_DIR_NAME,
    _get_dataset_arff,
    _get_dataset_description,
    _get_dataset_features_file,
    _get_dataset_parquet,
    _get_dataset_qualities_file,
    _get_online_dataset_arff,
    _get_online_dataset_format,
    _topic_add_dataset,
    _topic_delete_dataset,
    attributes_arff_from_df,
    create_dataset,
)
from openml.exceptions import (
    OpenMLHashException,
    OpenMLNotAuthorizedError,
    OpenMLPrivateDatasetError,
    OpenMLServerException,
)
from openml.tasks import TaskType, create_task
from openml.testing import TestBase, create_request_response
from openml.utils import _create_cache_directory_for_id, _tag_entity


class TestOpenMLDataset(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()

    def tearDown(self):
        self._remove_pickle_files()
        super().tearDown()

    def _remove_pickle_files(self):
        self.lock_path = os.path.join(openml.config.get_cache_directory(), "locks")
        for did in ["-1", "2"]:
            with lockutils.external_lock(
                name="datasets.functions.get_dataset:%s" % did,
                lock_path=self.lock_path,
            ):
                pickle_path = os.path.join(
                    openml.config.get_cache_directory(),
                    "datasets",
                    did,
                    "dataset.pkl.py3",
                )
                try:
                    os.remove(pickle_path)
                except (OSError, FileNotFoundError):
                    #  Replaced a bare except. Not sure why either of these would be acceptable.
                    pass

    def _get_empty_param_for_dataset(self):
        return {
            "name": None,
            "description": None,
            "creator": None,
            "contributor": None,
            "collection_date": None,
            "language": None,
            "licence": None,
            "default_target_attribute": None,
            "row_id_attribute": None,
            "ignore_attribute": None,
            "citation": None,
            "attributes": None,
            "data": None,
        }

    def _check_dataset(self, dataset):
        assert type(dataset) == dict
        assert len(dataset) >= 2
        assert "did" in dataset
        assert isinstance(dataset["did"], int)
        assert "status" in dataset
        assert isinstance(dataset["status"], str)
        assert dataset["status"] in ["in_preparation", "active", "deactivated"]

    def _check_datasets(self, datasets):
        for did in datasets:
            self._check_dataset(datasets[did])

    def test_tag_untag_dataset(self):
        tag = "test_tag_%d" % random.randint(1, 1000000)
        all_tags = _tag_entity("data", 1, tag)
        assert tag in all_tags
        all_tags = _tag_entity("data", 1, tag, untag=True)
        assert tag not in all_tags

    def test_list_datasets_output_format(self):
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        assert isinstance(datasets, pd.DataFrame)
        assert len(datasets) >= 100

    def test_list_datasets_paginate(self):
        size = 10
        max = 100
        for i in range(0, max, size):
            datasets = openml.datasets.list_datasets(offset=i, size=size)
            assert size == len(datasets)
            self._check_datasets(datasets)

    def test_list_datasets_empty(self):
        datasets = openml.datasets.list_datasets(
            tag="NoOneWouldUseThisTagAnyway",
            output_format="dataframe",
        )
        assert datasets.empty

    @pytest.mark.production()
    def test_check_datasets_active(self):
        # Have to test on live because there is no deactivated dataset on the test server.
        openml.config.server = self.production_server
        active = openml.datasets.check_datasets_active(
            [2, 17, 79],
            raise_error_if_not_exist=False,
        )
        assert active[2]
        assert not active[17]
        assert active.get(79) is None
        self.assertRaisesRegex(
            ValueError,
            r"Could not find dataset\(s\) 79 in OpenML dataset list.",
            openml.datasets.check_datasets_active,
            [79],
        )
        openml.config.server = self.test_server

    def test_illegal_character_tag(self):
        dataset = openml.datasets.get_dataset(1)
        tag = "illegal_tag&"
        try:
            dataset.push_tag(tag)
            raise AssertionError()
        except openml.exceptions.OpenMLServerException as e:
            assert e.code == 477

    def test_illegal_length_tag(self):
        dataset = openml.datasets.get_dataset(1)
        tag = "a" * 65
        try:
            dataset.push_tag(tag)
            raise AssertionError()
        except openml.exceptions.OpenMLServerException as e:
            assert e.code == 477

    def _datasets_retrieved_successfully(self, dids, metadata_only=True):
        """Checks that all files for the given dids have been downloaded.

        This includes:
            - description
            - qualities
            - features
            - absence of data arff if metadata_only, else it must be present too.
        """
        for did in dids:
            assert os.path.exists(
                os.path.join(
                    openml.config.get_cache_directory(), "datasets", str(did), "description.xml"
                )
            )
            assert os.path.exists(
                os.path.join(
                    openml.config.get_cache_directory(), "datasets", str(did), "qualities.xml"
                )
            )
            assert os.path.exists(
                os.path.join(
                    openml.config.get_cache_directory(), "datasets", str(did), "features.xml"
                )
            )

            data_assert = self.assertFalse if metadata_only else self.assertTrue
            data_assert(
                os.path.exists(
                    os.path.join(
                        openml.config.get_cache_directory(),
                        "datasets",
                        str(did),
                        "dataset.arff",
                    ),
                ),
            )

    @pytest.mark.production()
    def test__name_to_id_with_deactivated(self):
        """Check that an activated dataset is returned if an earlier deactivated one exists."""
        openml.config.server = self.production_server
        # /d/1 was deactivated
        assert openml.datasets.functions._name_to_id("anneal") == 2
        openml.config.server = self.test_server

    @pytest.mark.production()
    def test__name_to_id_with_multiple_active(self):
        """With multiple active datasets, retrieve the least recent active."""
        openml.config.server = self.production_server
        assert openml.datasets.functions._name_to_id("iris") == 61

    @pytest.mark.production()
    def test__name_to_id_with_version(self):
        """With multiple active datasets, retrieve the least recent active."""
        openml.config.server = self.production_server
        assert openml.datasets.functions._name_to_id("iris", version=3) == 969

    @pytest.mark.production()
    def test__name_to_id_with_multiple_active_error(self):
        """With multiple active datasets, retrieve the least recent active."""
        openml.config.server = self.production_server
        self.assertRaisesRegex(
            ValueError,
            "Multiple active datasets exist with name 'iris'.",
            openml.datasets.functions._name_to_id,
            dataset_name="iris",
            error_if_multiple=True,
        )

    def test__name_to_id_name_does_not_exist(self):
        """With multiple active datasets, retrieve the least recent active."""
        self.assertRaisesRegex(
            RuntimeError,
            "No active datasets exist with name 'does_not_exist'.",
            openml.datasets.functions._name_to_id,
            dataset_name="does_not_exist",
        )

    def test__name_to_id_version_does_not_exist(self):
        """With multiple active datasets, retrieve the least recent active."""
        self.assertRaisesRegex(
            RuntimeError,
            "No active datasets exist with name 'iris' and version '100000'.",
            openml.datasets.functions._name_to_id,
            dataset_name="iris",
            version=100000,
        )

    def test_get_datasets_by_name(self):
        # did 1 and 2 on the test server:
        dids = ["anneal", "kr-vs-kp"]
        datasets = openml.datasets.get_datasets(dids, download_data=False)
        assert len(datasets) == 2
        self._datasets_retrieved_successfully([1, 2])

    def test_get_datasets_by_mixed(self):
        # did 1 and 2 on the test server:
        dids = ["anneal", 2]
        datasets = openml.datasets.get_datasets(dids, download_data=False)
        assert len(datasets) == 2
        self._datasets_retrieved_successfully([1, 2])

    def test_get_datasets(self):
        dids = [1, 2]
        datasets = openml.datasets.get_datasets(dids)
        assert len(datasets) == 2
        self._datasets_retrieved_successfully([1, 2], metadata_only=False)

    def test_get_datasets_lazy(self):
        dids = [1, 2]
        datasets = openml.datasets.get_datasets(dids, download_data=False)
        assert len(datasets) == 2
        self._datasets_retrieved_successfully([1, 2], metadata_only=True)

        datasets[0].get_data()
        datasets[1].get_data()
        self._datasets_retrieved_successfully([1, 2], metadata_only=False)

    @pytest.mark.production()
    def test_get_dataset_by_name(self):
        dataset = openml.datasets.get_dataset("anneal")
        assert type(dataset) == OpenMLDataset
        assert dataset.dataset_id == 1
        self._datasets_retrieved_successfully([1], metadata_only=False)

        assert len(dataset.features) > 1
        assert len(dataset.qualities) > 4

        # Issue324 Properly handle private datasets when trying to access them
        openml.config.server = self.production_server
        self.assertRaises(OpenMLPrivateDatasetError, openml.datasets.get_dataset, 45)

    @pytest.mark.skip("Feature is experimental, can not test against stable server.")
    def test_get_dataset_download_all_files(self):
        # openml.datasets.get_dataset(id, download_all_files=True)
        # check for expected files
        # checking that no additional files are downloaded if
        # the default (false) is used, seems covered by
        # test_get_dataset_lazy
        raise NotImplementedError

    def test_get_dataset_uint8_dtype(self):
        dataset = openml.datasets.get_dataset(1)
        assert type(dataset) == OpenMLDataset
        assert dataset.name == "anneal"
        df, _, _, _ = dataset.get_data()
        assert df["carbon"].dtype == "uint8"

    @pytest.mark.production()
    def test_get_dataset(self):
        # This is the only non-lazy load to ensure default behaviour works.
        dataset = openml.datasets.get_dataset(1)
        assert type(dataset) == OpenMLDataset
        assert dataset.name == "anneal"
        self._datasets_retrieved_successfully([1], metadata_only=False)

        assert len(dataset.features) > 1
        assert len(dataset.qualities) > 4

        # Issue324 Properly handle private datasets when trying to access them
        openml.config.server = self.production_server
        self.assertRaises(OpenMLPrivateDatasetError, openml.datasets.get_dataset, 45)

    @pytest.mark.production()
    def test_get_dataset_lazy(self):
        dataset = openml.datasets.get_dataset(1, download_data=False)
        assert type(dataset) == OpenMLDataset
        assert dataset.name == "anneal"
        self._datasets_retrieved_successfully([1], metadata_only=True)

        assert len(dataset.features) > 1
        assert len(dataset.qualities) > 4

        dataset.get_data()
        self._datasets_retrieved_successfully([1], metadata_only=False)

        # Issue324 Properly handle private datasets when trying to access them
        openml.config.server = self.production_server
        self.assertRaises(OpenMLPrivateDatasetError, openml.datasets.get_dataset, 45, False)

    def test_get_dataset_lazy_all_functions(self):
        """Test that all expected functionality is available without downloading the dataset."""
        dataset = openml.datasets.get_dataset(1, download_data=False)
        # We only tests functions as general integrity is tested by test_get_dataset_lazy

        def ensure_absence_of_real_data():
            assert not os.path.exists(
                os.path.join(openml.config.get_cache_directory(), "datasets", "1", "dataset.arff")
            )

        tag = "test_lazy_tag_%d" % random.randint(1, 1000000)
        dataset.push_tag(tag)
        ensure_absence_of_real_data()

        dataset.remove_tag(tag)
        ensure_absence_of_real_data()

        nominal_indices = dataset.get_features_by_type("nominal")
        # fmt: off
        correct = [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38]
        # fmt: on
        assert nominal_indices == correct
        ensure_absence_of_real_data()

        classes = dataset.retrieve_class_labels()
        assert classes == ["1", "2", "3", "4", "5", "U"]
        ensure_absence_of_real_data()

    def test_get_dataset_sparse(self):
        dataset = openml.datasets.get_dataset(102, download_data=False)
        X, *_ = dataset.get_data(dataset_format="array")
        assert isinstance(X, scipy.sparse.csr_matrix)

    def test_download_rowid(self):
        # Smoke test which checks that the dataset has the row-id set correctly
        did = 44
        dataset = openml.datasets.get_dataset(did, download_data=False)
        assert dataset.row_id_attribute == "Counter"

    def test__get_dataset_description(self):
        description = _get_dataset_description(self.workdir, 2)
        assert isinstance(description, dict)
        description_xml_path = os.path.join(self.workdir, "description.xml")
        assert os.path.exists(description_xml_path)

    def test__getarff_path_dataset_arff(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        description = _get_dataset_description(self.workdir, 2)
        arff_path = _get_dataset_arff(description, cache_directory=self.workdir)
        assert isinstance(arff_path, Path)
        assert arff_path.exists()

    def test__download_minio_file_object_does_not_exist(self):
        self.assertRaisesRegex(
            FileNotFoundError,
            r"Object at .* does not exist",
            _download_minio_file,
            source="http://openml1.win.tue.nl/dataset20/i_do_not_exist.pq",
            destination=self.workdir,
            exists_ok=True,
        )

    def test__download_minio_file_to_directory(self):
        _download_minio_file(
            source="http://openml1.win.tue.nl/dataset20/dataset_20.pq",
            destination=self.workdir,
            exists_ok=True,
        )
        assert os.path.isfile(
            os.path.join(self.workdir, "dataset_20.pq")
        ), "_download_minio_file can save to a folder by copying the object name"

    def test__download_minio_file_to_path(self):
        file_destination = os.path.join(self.workdir, "custom.pq")
        _download_minio_file(
            source="http://openml1.win.tue.nl/dataset20/dataset_20.pq",
            destination=file_destination,
            exists_ok=True,
        )
        assert os.path.isfile(
            file_destination
        ), "_download_minio_file can save to a folder by copying the object name"

    def test__download_minio_file_raises_FileExists_if_destination_in_use(self):
        file_destination = Path(self.workdir, "custom.pq")
        file_destination.touch()

        self.assertRaises(
            FileExistsError,
            _download_minio_file,
            source="http://openml1.win.tue.nl/dataset20/dataset_20.pq",
            destination=str(file_destination),
            exists_ok=False,
        )

    def test__download_minio_file_works_with_bucket_subdirectory(self):
        file_destination = Path(self.workdir, "custom.pq")
        _download_minio_file(
            source="http://openml1.win.tue.nl/dataset61/dataset_61.pq",
            destination=file_destination,
            exists_ok=True,
        )
        assert os.path.isfile(
            file_destination
        ), "_download_minio_file can download from subdirectories"

    def test__get_dataset_parquet_not_cached(self):
        description = {
            "oml:parquet_url": "http://openml1.win.tue.nl/dataset20/dataset_20.pq",
            "oml:id": "20",
        }
        path = _get_dataset_parquet(description, cache_directory=self.workdir)
        assert isinstance(path, Path), "_get_dataset_parquet returns a path"
        assert path.is_file(), "_get_dataset_parquet returns path to real file"

    @mock.patch("openml._api_calls._download_minio_file")
    def test__get_dataset_parquet_is_cached(self, patch):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        patch.side_effect = RuntimeError(
            "_download_parquet_url should not be called when loading from cache",
        )
        description = {
            "oml:parquet_url": "http://openml1.win.tue.nl/dataset30/dataset_30.pq",
            "oml:id": "30",
        }
        path = _get_dataset_parquet(description, cache_directory=None)
        assert isinstance(path, Path), "_get_dataset_parquet returns a path"
        assert path.is_file(), "_get_dataset_parquet returns path to real file"

    def test__get_dataset_parquet_file_does_not_exist(self):
        description = {
            "oml:parquet_url": "http://openml1.win.tue.nl/dataset20/does_not_exist.pq",
            "oml:id": "20",
        }
        path = _get_dataset_parquet(description, cache_directory=self.workdir)
        assert path is None, "_get_dataset_parquet returns None if no file is found"

    def test__getarff_md5_issue(self):
        description = {
            "oml:id": 5,
            "oml:md5_checksum": "abc",
            "oml:url": "https://www.openml.org/data/download/61",
        }
        n = openml.config.connection_n_retries
        openml.config.connection_n_retries = 1

        self.assertRaisesRegex(
            OpenMLHashException,
            "Checksum of downloaded file is unequal to the expected checksum abc when downloading "
            "https://www.openml.org/data/download/61. Raised when downloading dataset 5.",
            _get_dataset_arff,
            description,
        )

        openml.config.connection_n_retries = n

    def test__get_dataset_features(self):
        features_file = _get_dataset_features_file(self.workdir, 2)
        assert isinstance(features_file, Path)
        features_xml_path = self.workdir / "features.xml"
        assert features_xml_path.exists()

    def test__get_dataset_qualities(self):
        qualities = _get_dataset_qualities_file(self.workdir, 2)
        assert isinstance(qualities, Path)
        qualities_xml_path = self.workdir / "qualities.xml"
        assert qualities_xml_path.exists()

    def test__get_dataset_skip_download(self):
        dataset = openml.datasets.get_dataset(
            2,
            download_qualities=False,
            download_features_meta_data=False,
        )
        # Internal representation without lazy loading
        assert dataset._qualities is None
        assert dataset._features is None
        # External representation with lazy loading
        assert dataset.qualities is not None
        assert dataset.features is not None

    def test_get_dataset_force_refresh_cache(self):
        did_cache_dir = _create_cache_directory_for_id(
            DATASETS_CACHE_DIR_NAME,
            2,
        )
        openml.datasets.get_dataset(2)
        change_time = os.stat(did_cache_dir).st_mtime

        # Test default
        openml.datasets.get_dataset(2)
        assert change_time == os.stat(did_cache_dir).st_mtime

        # Test refresh
        openml.datasets.get_dataset(2, force_refresh_cache=True)
        assert change_time != os.stat(did_cache_dir).st_mtime

        # Final clean up
        openml.utils._remove_cache_dir_for_id(
            DATASETS_CACHE_DIR_NAME,
            did_cache_dir,
        )

    def test_get_dataset_force_refresh_cache_clean_start(self):
        did_cache_dir = _create_cache_directory_for_id(
            DATASETS_CACHE_DIR_NAME,
            2,
        )
        # Clean up
        openml.utils._remove_cache_dir_for_id(
            DATASETS_CACHE_DIR_NAME,
            did_cache_dir,
        )

        # Test clean start
        openml.datasets.get_dataset(2, force_refresh_cache=True)
        assert os.path.exists(did_cache_dir)

        # Final clean up
        openml.utils._remove_cache_dir_for_id(
            DATASETS_CACHE_DIR_NAME,
            did_cache_dir,
        )

    def test_deletion_of_cache_dir(self):
        # Simple removal
        did_cache_dir = _create_cache_directory_for_id(
            DATASETS_CACHE_DIR_NAME,
            1,
        )
        assert os.path.exists(did_cache_dir)
        openml.utils._remove_cache_dir_for_id(
            DATASETS_CACHE_DIR_NAME,
            did_cache_dir,
        )
        assert not os.path.exists(did_cache_dir)

    # Use _get_dataset_arff to load the description, trigger an exception in the
    # test target and have a slightly higher coverage
    @mock.patch("openml.datasets.functions._get_dataset_arff")
    def test_deletion_of_cache_dir_faulty_download(self, patch):
        patch.side_effect = Exception("Boom!")
        self.assertRaisesRegex(Exception, "Boom!", openml.datasets.get_dataset, dataset_id=1)
        datasets_cache_dir = os.path.join(self.workdir, "org", "openml", "test", "datasets")
        assert len(os.listdir(datasets_cache_dir)) == 0

    def test_publish_dataset(self):
        # lazy loading not possible as we need the arff-file.
        openml.datasets.get_dataset(3)
        file_path = os.path.join(
            openml.config.get_cache_directory(),
            "datasets",
            "3",
            "dataset.arff",
        )
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
        TestBase._mark_entity_for_removal("data", dataset.dataset_id)
        TestBase.logger.info(
            "collected from {}: {}".format(__file__.split("/")[-1], dataset.dataset_id),
        )
        assert isinstance(dataset.dataset_id, int)

    def test__retrieve_class_labels(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        labels = openml.datasets.get_dataset(2, download_data=False).retrieve_class_labels()
        assert labels == ["1", "2", "3", "4", "5", "U"]

        labels = openml.datasets.get_dataset(2, download_data=False).retrieve_class_labels(
            target_name="product-type",
        )
        assert labels == ["C", "H", "G"]

        # Test workaround for string-typed class labels
        custom_ds = openml.datasets.get_dataset(2, download_data=False)
        custom_ds.features[31].data_type = "string"
        labels = custom_ds.retrieve_class_labels(target_name=custom_ds.features[31].name)
        assert labels == ["COIL", "SHEET"]

    def test_upload_dataset_with_url(self):
        dataset = OpenMLDataset(
            "%s-UploadTestWithURL" % self._get_sentinel(),
            "test",
            data_format="arff",
            version=1,
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff",
        )
        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.dataset_id)
        TestBase.logger.info(
            "collected from {}: {}".format(__file__.split("/")[-1], dataset.dataset_id),
        )
        assert isinstance(dataset.dataset_id, int)

    def _assert_status_of_dataset(self, *, did: int, status: str):
        """Asserts there is exactly one dataset with id `did` and its current status is `status`"""
        # need to use listing fn, as this is immune to cache
        result = openml.datasets.list_datasets(
            data_id=[did],
            status="all",
            output_format="dataframe",
        )
        result = result.to_dict(orient="index")
        # I think we should drop the test that one result is returned,
        # the server should never return multiple results?
        assert len(result) == 1
        assert result[did]["status"] == status

    @pytest.mark.flaky()
    def test_data_status(self):
        dataset = OpenMLDataset(
            "%s-UploadTestWithURL" % self._get_sentinel(),
            "test",
            "ARFF",
            version=1,
            url="https://www.openml.org/data/download/61/dataset_61_iris.arff",
        )
        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], dataset.id))
        did = dataset.id

        # admin key for test server (only adminds can activate datasets.
        # all users can deactivate their own datasets)
        openml.config.apikey = "d488d8afd93b32331cf6ea9d7003d4c3"

        openml.datasets.status_update(did, "active")
        self._assert_status_of_dataset(did=did, status="active")

        openml.datasets.status_update(did, "deactivated")
        self._assert_status_of_dataset(did=did, status="deactivated")

        openml.datasets.status_update(did, "active")
        self._assert_status_of_dataset(did=did, status="active")

        with pytest.raises(ValueError):
            openml.datasets.status_update(did, "in_preparation")
        self._assert_status_of_dataset(did=did, status="active")

    def test_attributes_arff_from_df(self):
        # DataFrame case
        df = pd.DataFrame(
            [[1, 1.0, "xxx", "A", True], [2, 2.0, "yyy", "B", False]],
            columns=["integer", "floating", "string", "category", "boolean"],
        )
        df["category"] = df["category"].astype("category")
        attributes = attributes_arff_from_df(df)
        assert attributes == [
            ("integer", "INTEGER"),
            ("floating", "REAL"),
            ("string", "STRING"),
            ("category", ["A", "B"]),
            ("boolean", ["True", "False"]),
        ]
        # DataFrame with Sparse columns case
        df = pd.DataFrame(
            {
                "integer": pd.arrays.SparseArray([1, 2, 0], fill_value=0),
                "floating": pd.arrays.SparseArray([1.0, 2.0, 0], fill_value=0.0),
            },
        )
        df["integer"] = df["integer"].astype(np.int64)
        attributes = attributes_arff_from_df(df)
        assert attributes == [("integer", "INTEGER"), ("floating", "REAL")]

    def test_attributes_arff_from_df_numeric_column(self):
        # Test column names are automatically converted to str if needed (#819)
        df = pd.DataFrame({0: [1, 2, 3], 0.5: [4, 5, 6], "target": [0, 1, 1]})
        attributes = attributes_arff_from_df(df)
        assert attributes == [("0", "INTEGER"), ("0.5", "INTEGER"), ("target", "INTEGER")]

    def test_attributes_arff_from_df_mixed_dtype_categories(self):
        # liac-arff imposed categorical attributes to be of sting dtype. We
        # raise an error if this is not the case.
        df = pd.DataFrame([[1], ["2"], [3.0]])
        df[0] = df[0].astype("category")
        err_msg = "The column '0' of the dataframe is of 'category' dtype."
        with pytest.raises(ValueError, match=err_msg):
            attributes_arff_from_df(df)

    def test_attributes_arff_from_df_unknown_dtype(self):
        # check that an error is raised when the dtype is not supptagorted by
        # liac-arff
        data = [
            [[1], ["2"], [3.0]],
            [pd.Timestamp("2012-05-01"), pd.Timestamp("2012-05-02")],
        ]
        dtype = ["mixed-integer", "datetime64"]
        for arr, dt in zip(data, dtype):
            df = pd.DataFrame(arr)
            err_msg = (
                f"The dtype '{dt}' of the column '0' is not currently " "supported by liac-arff"
            )
            with pytest.raises(ValueError, match=err_msg):
                attributes_arff_from_df(df)

    def test_create_dataset_numpy(self):
        data = np.array([[1, 2, 3], [1.2, 2.5, 3.8], [2, 5, 8], [0, 1, 0]]).T

        attributes = [(f"col_{i}", "REAL") for i in range(data.shape[1])]

        dataset = create_dataset(
            name="%s-NumPy_testing_dataset" % self._get_sentinel(),
            description="Synthetic dataset created from a NumPy array",
            creator="OpenML tester",
            contributor=None,
            collection_date="01-01-2018",
            language="English",
            licence="MIT",
            default_target_attribute=f"col_{data.shape[1] - 1}",
            row_id_attribute=None,
            ignore_attribute=None,
            citation="None",
            attributes=attributes,
            data=data,
            version_label="test",
            original_data_url="http://openml.github.io/openml-python",
            paper_url="http://openml.github.io/openml-python",
        )

        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], dataset.id))

        assert (
            _get_online_dataset_arff(dataset.id) == dataset._dataset
        ), "Uploaded arff does not match original one"
        assert _get_online_dataset_format(dataset.id) == "arff", "Wrong format for dataset"

    def test_create_dataset_list(self):
        data = [
            ["a", "sunny", 85.0, 85.0, "FALSE", "no"],
            ["b", "sunny", 80.0, 90.0, "TRUE", "no"],
            ["c", "overcast", 83.0, 86.0, "FALSE", "yes"],
            ["d", "rainy", 70.0, 96.0, "FALSE", "yes"],
            ["e", "rainy", 68.0, 80.0, "FALSE", "yes"],
            ["f", "rainy", 65.0, 70.0, "TRUE", "no"],
            ["g", "overcast", 64.0, 65.0, "TRUE", "yes"],
            ["h", "sunny", 72.0, 95.0, "FALSE", "no"],
            ["i", "sunny", 69.0, 70.0, "FALSE", "yes"],
            ["j", "rainy", 75.0, 80.0, "FALSE", "yes"],
            ["k", "sunny", 75.0, 70.0, "TRUE", "yes"],
            ["l", "overcast", 72.0, 90.0, "TRUE", "yes"],
            ["m", "overcast", 81.0, 75.0, "FALSE", "yes"],
            ["n", "rainy", 71.0, 91.0, "TRUE", "no"],
        ]

        attributes = [
            ("rnd_str", "STRING"),
            ("outlook", ["sunny", "overcast", "rainy"]),
            ("temperature", "REAL"),
            ("humidity", "REAL"),
            ("windy", ["TRUE", "FALSE"]),
            ("play", ["yes", "no"]),
        ]

        dataset = create_dataset(
            name="%s-ModifiedWeather" % self._get_sentinel(),
            description=("Testing dataset upload when the data is a list of lists"),
            creator="OpenML test",
            contributor=None,
            collection_date="21-09-2018",
            language="English",
            licence="MIT",
            default_target_attribute="play",
            row_id_attribute=None,
            ignore_attribute=None,
            citation="None",
            attributes=attributes,
            data=data,
            version_label="test",
            original_data_url="http://openml.github.io/openml-python",
            paper_url="http://openml.github.io/openml-python",
        )

        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], dataset.id))
        assert (
            _get_online_dataset_arff(dataset.id) == dataset._dataset
        ), "Uploaded ARFF does not match original one"
        assert _get_online_dataset_format(dataset.id) == "arff", "Wrong format for dataset"

    def test_create_dataset_sparse(self):
        # test the scipy.sparse.coo_matrix
        sparse_data = scipy.sparse.coo_matrix(
            ([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ([0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1])),
        )

        column_names = [
            ("input1", "REAL"),
            ("input2", "REAL"),
            ("y", "REAL"),
        ]

        xor_dataset = create_dataset(
            name="%s-XOR" % self._get_sentinel(),
            description="Dataset representing the XOR operation",
            creator=None,
            contributor=None,
            collection_date=None,
            language="English",
            licence=None,
            default_target_attribute="y",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=None,
            attributes=column_names,
            data=sparse_data,
            version_label="test",
        )

        xor_dataset.publish()
        TestBase._mark_entity_for_removal("data", xor_dataset.id)
        TestBase.logger.info(
            "collected from {}: {}".format(__file__.split("/")[-1], xor_dataset.id),
        )
        assert (
            _get_online_dataset_arff(xor_dataset.id) == xor_dataset._dataset
        ), "Uploaded ARFF does not match original one"
        assert (
            _get_online_dataset_format(xor_dataset.id) == "sparse_arff"
        ), "Wrong format for dataset"

        # test the list of dicts sparse representation
        sparse_data = [{0: 0.0}, {1: 1.0, 2: 1.0}, {0: 1.0, 2: 1.0}, {0: 1.0, 1: 1.0}]

        xor_dataset = create_dataset(
            name="%s-XOR" % self._get_sentinel(),
            description="Dataset representing the XOR operation",
            creator=None,
            contributor=None,
            collection_date=None,
            language="English",
            licence=None,
            default_target_attribute="y",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=None,
            attributes=column_names,
            data=sparse_data,
            version_label="test",
        )

        xor_dataset.publish()
        TestBase._mark_entity_for_removal("data", xor_dataset.id)
        TestBase.logger.info(
            "collected from {}: {}".format(__file__.split("/")[-1], xor_dataset.id),
        )
        assert (
            _get_online_dataset_arff(xor_dataset.id) == xor_dataset._dataset
        ), "Uploaded ARFF does not match original one"
        assert (
            _get_online_dataset_format(xor_dataset.id) == "sparse_arff"
        ), "Wrong format for dataset"

    def test_create_invalid_dataset(self):
        data = [
            "sunny",
            "overcast",
            "overcast",
            "rainy",
            "rainy",
            "rainy",
            "overcast",
            "sunny",
            "sunny",
            "rainy",
            "sunny",
            "overcast",
            "overcast",
            "rainy",
        ]

        param = self._get_empty_param_for_dataset()
        param["data"] = data

        self.assertRaises(ValueError, create_dataset, **param)

        param["data"] = data[0]
        self.assertRaises(ValueError, create_dataset, **param)

    def test_get_online_dataset_arff(self):
        dataset_id = 100  # Australian
        # lazy loading not used as arff file is checked.
        dataset = openml.datasets.get_dataset(dataset_id)
        decoder = arff.ArffDecoder()
        # check if the arff from the dataset is
        # the same as the arff from _get_arff function
        d_format = (dataset.format).lower()

        assert dataset._get_arff(d_format) == decoder.decode(
            _get_online_dataset_arff(dataset_id),
            encode_nominal=True,
            return_type=arff.DENSE if d_format == "arff" else arff.COO,
        ), "ARFF files are not equal"

    def test_topic_api_error(self):
        # Check server exception when non-admin accessses apis
        self.assertRaisesRegex(
            OpenMLServerException,
            "Topic can only be added/removed by admin.",
            _topic_add_dataset,
            data_id=31,
            topic="business",
        )
        # Check server exception when non-admin accessses apis
        self.assertRaisesRegex(
            OpenMLServerException,
            "Topic can only be added/removed by admin.",
            _topic_delete_dataset,
            data_id=31,
            topic="business",
        )

    def test_get_online_dataset_format(self):
        # Phoneme dataset
        dataset_id = 77
        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)

        assert dataset.format.lower() == _get_online_dataset_format(
            dataset_id
        ), "The format of the ARFF files is different"

    def test_create_dataset_pandas(self):
        data = [
            ["a", "sunny", 85.0, 85.0, "FALSE", "no"],
            ["b", "sunny", 80.0, 90.0, "TRUE", "no"],
            ["c", "overcast", 83.0, 86.0, "FALSE", "yes"],
            ["d", "rainy", 70.0, 96.0, "FALSE", "yes"],
            ["e", "rainy", 68.0, 80.0, "FALSE", "yes"],
        ]
        column_names = ["rnd_str", "outlook", "temperature", "humidity", "windy", "play"]
        df = pd.DataFrame(data, columns=column_names)
        # enforce the type of each column
        df["outlook"] = df["outlook"].astype("category")
        df["windy"] = df["windy"].astype("bool")
        df["play"] = df["play"].astype("category")
        # meta-information
        name = "%s-pandas_testing_dataset" % self._get_sentinel()
        description = "Synthetic dataset created from a Pandas DataFrame"
        creator = "OpenML tester"
        collection_date = "01-01-2018"
        language = "English"
        licence = "MIT"
        citation = "None"
        original_data_url = "http://openml.github.io/openml-python"
        paper_url = "http://openml.github.io/openml-python"
        dataset = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute="play",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=citation,
            attributes="auto",
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )
        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], dataset.id))
        assert (
            _get_online_dataset_arff(dataset.id) == dataset._dataset
        ), "Uploaded ARFF does not match original one"

        # Check that DataFrame with Sparse columns are supported properly
        sparse_data = scipy.sparse.coo_matrix(
            ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ([0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1])),
        )
        column_names = ["input1", "input2", "y"]
        df = pd.DataFrame.sparse.from_spmatrix(sparse_data, columns=column_names)
        # meta-information
        description = "Synthetic dataset created from a Pandas DataFrame with Sparse columns"
        dataset = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute="y",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=citation,
            attributes="auto",
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )
        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], dataset.id))
        assert (
            _get_online_dataset_arff(dataset.id) == dataset._dataset
        ), "Uploaded ARFF does not match original one"
        assert _get_online_dataset_format(dataset.id) == "sparse_arff", "Wrong format for dataset"

        # Check that we can overwrite the attributes
        data = [["a"], ["b"], ["c"], ["d"], ["e"]]
        column_names = ["rnd_str"]
        df = pd.DataFrame(data, columns=column_names)
        df["rnd_str"] = df["rnd_str"].astype("category")
        attributes = {"rnd_str": ["a", "b", "c", "d", "e", "f", "g"]}
        dataset = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute="rnd_str",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=citation,
            attributes=attributes,
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )
        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], dataset.id))
        downloaded_data = _get_online_dataset_arff(dataset.id)
        assert downloaded_data == dataset._dataset, "Uploaded ARFF does not match original one"
        assert "@ATTRIBUTE rnd_str {a, b, c, d, e, f, g}" in downloaded_data

    def test_ignore_attributes_dataset(self):
        data = [
            ["a", "sunny", 85.0, 85.0, "FALSE", "no"],
            ["b", "sunny", 80.0, 90.0, "TRUE", "no"],
            ["c", "overcast", 83.0, 86.0, "FALSE", "yes"],
            ["d", "rainy", 70.0, 96.0, "FALSE", "yes"],
            ["e", "rainy", 68.0, 80.0, "FALSE", "yes"],
        ]
        column_names = ["rnd_str", "outlook", "temperature", "humidity", "windy", "play"]
        df = pd.DataFrame(data, columns=column_names)
        # enforce the type of each column
        df["outlook"] = df["outlook"].astype("category")
        df["windy"] = df["windy"].astype("bool")
        df["play"] = df["play"].astype("category")
        # meta-information
        name = "%s-pandas_testing_dataset" % self._get_sentinel()
        description = "Synthetic dataset created from a Pandas DataFrame"
        creator = "OpenML tester"
        collection_date = "01-01-2018"
        language = "English"
        licence = "MIT"
        default_target_attribute = "play"
        citation = "None"
        original_data_url = "http://openml.github.io/openml-python"
        paper_url = "http://openml.github.io/openml-python"

        # we use the create_dataset function which call the OpenMLDataset
        # constructor
        # pass a string to ignore_attribute
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
            ignore_attribute="outlook",
            citation=citation,
            attributes="auto",
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )
        assert dataset.ignore_attribute == ["outlook"]

        # pass a list to ignore_attribute
        ignore_attribute = ["outlook", "windy"]
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
            ignore_attribute=ignore_attribute,
            citation=citation,
            attributes="auto",
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )
        assert dataset.ignore_attribute == ignore_attribute

        # raise an error if unknown type
        err_msg = "Wrong data type for ignore_attribute. Should be list."
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
                ignore_attribute=("outlook", "windy"),
                citation=citation,
                attributes="auto",
                data=df,
                version_label="test",
                original_data_url=original_data_url,
                paper_url=paper_url,
            )

    def test_publish_fetch_ignore_attribute(self):
        """Test to upload and retrieve dataset and check ignore_attributes"""
        data = [
            ["a", "sunny", 85.0, 85.0, "FALSE", "no"],
            ["b", "sunny", 80.0, 90.0, "TRUE", "no"],
            ["c", "overcast", 83.0, 86.0, "FALSE", "yes"],
            ["d", "rainy", 70.0, 96.0, "FALSE", "yes"],
            ["e", "rainy", 68.0, 80.0, "FALSE", "yes"],
        ]
        column_names = ["rnd_str", "outlook", "temperature", "humidity", "windy", "play"]
        df = pd.DataFrame(data, columns=column_names)
        # enforce the type of each column
        df["outlook"] = df["outlook"].astype("category")
        df["windy"] = df["windy"].astype("bool")
        df["play"] = df["play"].astype("category")
        # meta-information
        name = "%s-pandas_testing_dataset" % self._get_sentinel()
        description = "Synthetic dataset created from a Pandas DataFrame"
        creator = "OpenML tester"
        collection_date = "01-01-2018"
        language = "English"
        licence = "MIT"
        default_target_attribute = "play"
        citation = "None"
        original_data_url = "http://openml.github.io/openml-python"
        paper_url = "http://openml.github.io/openml-python"

        # pass a list to ignore_attribute
        ignore_attribute = ["outlook", "windy"]
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
            ignore_attribute=ignore_attribute,
            citation=citation,
            attributes="auto",
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )

        # publish dataset
        dataset.publish()
        TestBase._mark_entity_for_removal("data", dataset.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], dataset.id))
        # test if publish was successful
        assert isinstance(dataset.id, int)

        downloaded_dataset = self._wait_for_dataset_being_processed(dataset.id)
        assert downloaded_dataset.ignore_attribute == ignore_attribute

    def _wait_for_dataset_being_processed(self, dataset_id):
        downloaded_dataset = None
        # fetching from server
        # loop till timeout or fetch not successful
        max_waiting_time_seconds = 600
        # time.time() works in seconds
        start_time = time.time()
        while time.time() - start_time < max_waiting_time_seconds:
            try:
                downloaded_dataset = openml.datasets.get_dataset(dataset_id)
                break
            except OpenMLServerException as e:
                # returned code 273: Dataset not processed yet
                # returned code 362: No qualities found
                TestBase.logger.error(
                    f"Failed to fetch dataset:{dataset_id} with '{e!s}'.",
                )
                time.sleep(10)
                continue
        if downloaded_dataset is None:
            raise ValueError(f"TIMEOUT: Failed to fetch uploaded dataset - {dataset_id}")
        return downloaded_dataset

    def test_create_dataset_row_id_attribute_error(self):
        # meta-information
        name = "%s-pandas_testing_dataset" % self._get_sentinel()
        description = "Synthetic dataset created from a Pandas DataFrame"
        creator = "OpenML tester"
        collection_date = "01-01-2018"
        language = "English"
        licence = "MIT"
        default_target_attribute = "target"
        citation = "None"
        original_data_url = "http://openml.github.io/openml-python"
        paper_url = "http://openml.github.io/openml-python"
        # Check that the index name is well inferred.
        data = [["a", 1, 0], ["b", 2, 1], ["c", 3, 0], ["d", 4, 1], ["e", 5, 0]]
        column_names = ["rnd_str", "integer", "target"]
        df = pd.DataFrame(data, columns=column_names)
        # affecting row_id_attribute to an unknown column should raise an error
        err_msg = "should be one of the data attribute."
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
                attributes="auto",
                data=df,
                row_id_attribute="unknown_row_id",
                version_label="test",
                original_data_url=original_data_url,
                paper_url=paper_url,
            )

    def test_create_dataset_row_id_attribute_inference(self):
        # meta-information
        name = "%s-pandas_testing_dataset" % self._get_sentinel()
        description = "Synthetic dataset created from a Pandas DataFrame"
        creator = "OpenML tester"
        collection_date = "01-01-2018"
        language = "English"
        licence = "MIT"
        default_target_attribute = "target"
        citation = "None"
        original_data_url = "http://openml.github.io/openml-python"
        paper_url = "http://openml.github.io/openml-python"
        # Check that the index name is well inferred.
        data = [["a", 1, 0], ["b", 2, 1], ["c", 3, 0], ["d", 4, 1], ["e", 5, 0]]
        column_names = ["rnd_str", "integer", "target"]
        df = pd.DataFrame(data, columns=column_names)
        row_id_attr = [None, "integer"]
        df_index_name = [None, "index_name"]
        expected_row_id = [None, "index_name", "integer", "integer"]
        for output_row_id, (row_id, index_name) in zip(
            expected_row_id,
            product(row_id_attr, df_index_name),
        ):
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
                attributes="auto",
                data=df,
                row_id_attribute=row_id,
                version_label="test",
                original_data_url=original_data_url,
                paper_url=paper_url,
            )
            assert dataset.row_id_attribute == output_row_id
            dataset.publish()
            TestBase._mark_entity_for_removal("data", dataset.id)
            TestBase.logger.info(
                "collected from {}: {}".format(__file__.split("/")[-1], dataset.id),
            )
            arff_dataset = arff.loads(_get_online_dataset_arff(dataset.id))
            arff_data = np.array(arff_dataset["data"], dtype=object)
            # if we set the name of the index then the index will be added to
            # the data
            expected_shape = (5, 3) if index_name is None else (5, 4)
            assert arff_data.shape == expected_shape

    def test_create_dataset_attributes_auto_without_df(self):
        # attributes cannot be inferred without passing a dataframe
        data = np.array([[1, 2, 3], [1.2, 2.5, 3.8], [2, 5, 8], [0, 1, 0]]).T
        attributes = "auto"
        name = "NumPy_testing_dataset"
        description = "Synthetic dataset created from a NumPy array"
        creator = "OpenML tester"
        collection_date = "01-01-2018"
        language = "English"
        licence = "MIT"
        default_target_attribute = f"col_{data.shape[1] - 1}"
        citation = "None"
        original_data_url = "http://openml.github.io/openml-python"
        paper_url = "http://openml.github.io/openml-python"
        err_msg = "Automatically inferring attributes requires a pandas"
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
                version_label="test",
                original_data_url=original_data_url,
                paper_url=paper_url,
            )

    def test_list_qualities(self):
        qualities = openml.datasets.list_qualities()
        assert isinstance(qualities, list) is True
        assert all(isinstance(q, str) for q in qualities) is True

    def test_get_dataset_cache_format_pickle(self):
        dataset = openml.datasets.get_dataset(1)
        dataset.get_data()

        assert type(dataset) == OpenMLDataset
        assert dataset.name == "anneal"
        assert len(dataset.features) > 1
        assert len(dataset.qualities) > 4

        X, y, categorical, attribute_names = dataset.get_data()
        assert isinstance(X, pd.DataFrame)
        assert X.shape == (898, 39)
        assert len(categorical) == X.shape[1]
        assert len(attribute_names) == X.shape[1]

    def test_get_dataset_cache_format_feather(self):
        # This test crashed due to using the parquet file by default, which is downloaded
        # from minio. However, there is a mismatch between OpenML test server and minio IDs.
        # The parquet file on minio with ID 128 is not the iris dataset from the test server.
        dataset = openml.datasets.get_dataset(128, cache_format="feather")
        # Workaround
        dataset._parquet_url = None
        dataset.parquet_file = None
        dataset.get_data()

        # Check if dataset is written to cache directory using feather
        cache_dir = openml.config.get_cache_directory()
        cache_dir_for_id = os.path.join(cache_dir, "datasets", "128")
        feather_file = os.path.join(cache_dir_for_id, "dataset.feather")
        pickle_file = os.path.join(cache_dir_for_id, "dataset.feather.attributes.pkl.py3")
        data = pd.read_feather(feather_file)
        assert os.path.isfile(feather_file), "Feather file is missing"
        assert os.path.isfile(pickle_file), "Attributes pickle file is missing"
        assert data.shape == (150, 5)

        # Check if get_data is able to retrieve feather data
        assert type(dataset) == OpenMLDataset
        assert dataset.name == "iris"
        assert len(dataset.features) > 1
        assert len(dataset.qualities) > 4

        X, y, categorical, attribute_names = dataset.get_data()
        assert isinstance(X, pd.DataFrame)
        assert X.shape == (150, 5)
        assert len(categorical) == X.shape[1]
        assert len(attribute_names) == X.shape[1]

    def test_data_edit_non_critical_field(self):
        # Case 1
        # All users can edit non-critical fields of datasets
        desc = (
            "This data sets consists of 3 different types of irises' "
            "(Setosa, Versicolour, and Virginica) petal and sepal length,"
            " stored in a 150x4 numpy.ndarray"
        )
        did = 128
        result = edit_dataset(
            did,
            description=desc,
            creator="R.A.Fisher",
            collection_date="1937",
            citation="The use of multiple measurements in taxonomic problems",
            language="English",
        )
        assert did == result
        edited_dataset = openml.datasets.get_dataset(did)
        assert edited_dataset.description == desc

    def test_data_edit_critical_field(self):
        # Case 2
        # only owners (or admin) can edit all critical fields of datasets
        # for this, we need to first clone a dataset to do changes
        did = fork_dataset(1)
        self._wait_for_dataset_being_processed(did)
        result = edit_dataset(did, default_target_attribute="shape", ignore_attribute="oil")
        assert did == result

        n_tries = 10
        # we need to wait for the edit to be reflected on the server
        for i in range(n_tries):
            edited_dataset = openml.datasets.get_dataset(did)
            try:
                assert edited_dataset.default_target_attribute == "shape", edited_dataset
                assert edited_dataset.ignore_attribute == ["oil"], edited_dataset
                break
            except AssertionError as e:
                if i == n_tries - 1:
                    raise e
                time.sleep(10)
                # Delete the cache dir to get the newer version of the dataset
                shutil.rmtree(
                    os.path.join(self.workdir, "org", "openml", "test", "datasets", str(did)),
                )

    def test_data_edit_errors(self):
        # Check server exception when no field to edit is provided
        self.assertRaisesRegex(
            OpenMLServerException,
            "Please provide atleast one field among description, creator, "
            "contributor, collection_date, language, citation, "
            "original_data_url, default_target_attribute, row_id_attribute, "
            "ignore_attribute or paper_url to edit.",
            edit_dataset,
            data_id=64,  # blood-transfusion-service-center
        )
        # Check server exception when unknown dataset is provided
        self.assertRaisesRegex(
            OpenMLServerException,
            "Unknown dataset",
            edit_dataset,
            data_id=999999,
            description="xor operation dataset",
        )

        # Need to own a dataset to be able to edit meta-data
        # Will be creating a forked version of an existing dataset to allow the unit test user
        #  to edit meta-data of a dataset
        did = fork_dataset(1)
        self._wait_for_dataset_being_processed(did)
        TestBase._mark_entity_for_removal("data", did)
        # Need to upload a task attached to this data to test edit failure
        task = create_task(
            task_type=TaskType.SUPERVISED_CLASSIFICATION,
            dataset_id=did,
            target_name="class",
            estimation_procedure_id=1,
        )
        task = task.publish()
        TestBase._mark_entity_for_removal("task", task.task_id)
        # Check server exception when owner/admin edits critical fields of dataset with tasks
        self.assertRaisesRegex(
            OpenMLServerException,
            "Critical features default_target_attribute, row_id_attribute and ignore_attribute "
            "can only be edited for datasets without any tasks.",
            edit_dataset,
            data_id=did,
            default_target_attribute="y",
        )

        # Check server exception when a non-owner or non-admin tries to edit critical fields
        self.assertRaisesRegex(
            OpenMLServerException,
            "Critical features default_target_attribute, row_id_attribute and ignore_attribute "
            "can be edited only by the owner. Fork the dataset if changes are required.",
            edit_dataset,
            data_id=128,
            default_target_attribute="y",
        )

    def test_data_fork(self):
        did = 1
        result = fork_dataset(did)
        assert did != result
        # Check server exception when unknown dataset is provided
        self.assertRaisesRegex(
            OpenMLServerException,
            "Unknown dataset",
            fork_dataset,
            data_id=999999,
        )

    @pytest.mark.production()
    def test_get_dataset_parquet(self):
        # Parquet functionality is disabled on the test server
        # There is no parquet-copy of the test server yet.
        openml.config.server = self.production_server
        dataset = openml.datasets.get_dataset(61)
        assert dataset._parquet_url is not None
        assert dataset.parquet_file is not None
        assert os.path.isfile(dataset.parquet_file)

    @pytest.mark.production()
    def test_list_datasets_with_high_size_parameter(self):
        # Testing on prod since concurrent deletion of uploded datasets make the test fail
        openml.config.server = self.production_server

        datasets_a = openml.datasets.list_datasets(output_format="dataframe")
        datasets_b = openml.datasets.list_datasets(output_format="dataframe", size=np.inf)

        # Reverting to test server
        openml.config.server = self.test_server
        assert len(datasets_a) == len(datasets_b)


@pytest.mark.parametrize(
    ("default_target_attribute", "row_id_attribute", "ignore_attribute"),
    [
        ("wrong", None, None),
        (None, "wrong", None),
        (None, None, "wrong"),
        ("wrong,sunny", None, None),
        (None, None, "wrong,sunny"),
        (["wrong", "sunny"], None, None),
        (None, None, ["wrong", "sunny"]),
    ],
)
def test_invalid_attribute_validations(
    default_target_attribute,
    row_id_attribute,
    ignore_attribute,
):
    data = [
        ["a", "sunny", 85.0, 85.0, "FALSE", "no"],
        ["b", "sunny", 80.0, 90.0, "TRUE", "no"],
        ["c", "overcast", 83.0, 86.0, "FALSE", "yes"],
        ["d", "rainy", 70.0, 96.0, "FALSE", "yes"],
        ["e", "rainy", 68.0, 80.0, "FALSE", "yes"],
    ]
    column_names = ["rnd_str", "outlook", "temperature", "humidity", "windy", "play"]
    df = pd.DataFrame(data, columns=column_names)
    # enforce the type of each column
    df["outlook"] = df["outlook"].astype("category")
    df["windy"] = df["windy"].astype("bool")
    df["play"] = df["play"].astype("category")
    # meta-information
    name = "pandas_testing_dataset"
    description = "Synthetic dataset created from a Pandas DataFrame"
    creator = "OpenML tester"
    collection_date = "01-01-2018"
    language = "English"
    licence = "MIT"
    citation = "None"
    original_data_url = "http://openml.github.io/openml-python"
    paper_url = "http://openml.github.io/openml-python"
    with pytest.raises(ValueError, match="should be one of the data attribute"):
        _ = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute=default_target_attribute,
            row_id_attribute=row_id_attribute,
            ignore_attribute=ignore_attribute,
            citation=citation,
            attributes="auto",
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )


@pytest.mark.parametrize(
    ("default_target_attribute", "row_id_attribute", "ignore_attribute"),
    [
        ("outlook", None, None),
        (None, "outlook", None),
        (None, None, "outlook"),
        ("outlook,windy", None, None),
        (None, None, "outlook,windy"),
        (["outlook", "windy"], None, None),
        (None, None, ["outlook", "windy"]),
    ],
)
def test_valid_attribute_validations(default_target_attribute, row_id_attribute, ignore_attribute):
    data = [
        ["a", "sunny", 85.0, 85.0, "FALSE", "no"],
        ["b", "sunny", 80.0, 90.0, "TRUE", "no"],
        ["c", "overcast", 83.0, 86.0, "FALSE", "yes"],
        ["d", "rainy", 70.0, 96.0, "FALSE", "yes"],
        ["e", "rainy", 68.0, 80.0, "FALSE", "yes"],
    ]
    column_names = ["rnd_str", "outlook", "temperature", "humidity", "windy", "play"]
    df = pd.DataFrame(data, columns=column_names)
    # enforce the type of each column
    df["outlook"] = df["outlook"].astype("category")
    df["windy"] = df["windy"].astype("bool")
    df["play"] = df["play"].astype("category")
    # meta-information
    name = "pandas_testing_dataset"
    description = "Synthetic dataset created from a Pandas DataFrame"
    creator = "OpenML tester"
    collection_date = "01-01-2018"
    language = "English"
    licence = "MIT"
    citation = "None"
    original_data_url = "http://openml.github.io/openml-python"
    paper_url = "http://openml.github.io/openml-python"
    _ = openml.datasets.functions.create_dataset(
        name=name,
        description=description,
        creator=creator,
        contributor=None,
        collection_date=collection_date,
        language=language,
        licence=licence,
        default_target_attribute=default_target_attribute,
        row_id_attribute=row_id_attribute,
        ignore_attribute=ignore_attribute,
        citation=citation,
        attributes="auto",
        data=df,
        version_label="test",
        original_data_url=original_data_url,
        paper_url=paper_url,
    )

    def test_delete_dataset(self):
        data = [
            ["a", "sunny", 85.0, 85.0, "FALSE", "no"],
            ["b", "sunny", 80.0, 90.0, "TRUE", "no"],
            ["c", "overcast", 83.0, 86.0, "FALSE", "yes"],
            ["d", "rainy", 70.0, 96.0, "FALSE", "yes"],
            ["e", "rainy", 68.0, 80.0, "FALSE", "yes"],
        ]
        column_names = ["rnd_str", "outlook", "temperature", "humidity", "windy", "play"]
        df = pd.DataFrame(data, columns=column_names)
        # enforce the type of each column
        df["outlook"] = df["outlook"].astype("category")
        df["windy"] = df["windy"].astype("bool")
        df["play"] = df["play"].astype("category")
        # meta-information
        name = "%s-pandas_testing_dataset" % self._get_sentinel()
        description = "Synthetic dataset created from a Pandas DataFrame"
        creator = "OpenML tester"
        collection_date = "01-01-2018"
        language = "English"
        licence = "MIT"
        citation = "None"
        original_data_url = "http://openml.github.io/openml-python"
        paper_url = "http://openml.github.io/openml-python"
        dataset = openml.datasets.functions.create_dataset(
            name=name,
            description=description,
            creator=creator,
            contributor=None,
            collection_date=collection_date,
            language=language,
            licence=licence,
            default_target_attribute="play",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=citation,
            attributes="auto",
            data=df,
            version_label="test",
            original_data_url=original_data_url,
            paper_url=paper_url,
        )
        dataset.publish()
        _dataset_id = dataset.id
        assert openml.datasets.delete_dataset(_dataset_id)


@mock.patch.object(requests.Session, "delete")
def test_delete_dataset_not_owned(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = (
        test_files_directory / "mock_responses" / "datasets" / "data_delete_not_owned.xml"
    )
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The data can not be deleted because it was not uploaded by you.",
    ):
        openml.datasets.delete_dataset(40_000)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/data/40000",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_dataset_with_run(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = (
        test_files_directory / "mock_responses" / "datasets" / "data_delete_has_tasks.xml"
    )
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The data can not be deleted because it still has associated entities:",
    ):
        openml.datasets.delete_dataset(40_000)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/data/40000",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_dataset_success(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = (
        test_files_directory / "mock_responses" / "datasets" / "data_delete_successful.xml"
    )
    mock_delete.return_value = create_request_response(
        status_code=200,
        content_filepath=content_file,
    )

    success = openml.datasets.delete_dataset(40000)
    assert success

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/data/40000",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_unknown_dataset(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = (
        test_files_directory / "mock_responses" / "datasets" / "data_delete_not_exist.xml"
    )
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLServerException,
        match="Dataset does not exist",
    ):
        openml.datasets.delete_dataset(9_999_999)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/data/9999999",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


def _assert_datasets_have_id_and_valid_status(datasets: pd.DataFrame):
    assert pd.api.types.is_integer_dtype(datasets["did"])
    assert {"in_preparation", "active", "deactivated"} >= set(datasets["status"])


@pytest.fixture(scope="module")
def all_datasets():
    return openml.datasets.list_datasets(output_format="dataframe")


def test_list_datasets(all_datasets: pd.DataFrame):
    # We can only perform a smoke test here because we test on dynamic
    # data from the internet...
    # 1087 as the number of datasets on openml.org
    assert len(all_datasets) >= 100
    _assert_datasets_have_id_and_valid_status(all_datasets)


def test_list_datasets_by_tag(all_datasets: pd.DataFrame):
    tag_datasets = openml.datasets.list_datasets(tag="study_14", output_format="dataframe")
    assert 0 < len(tag_datasets) < len(all_datasets)
    _assert_datasets_have_id_and_valid_status(tag_datasets)


def test_list_datasets_by_size():
    datasets = openml.datasets.list_datasets(size=5, output_format="dataframe")
    assert len(datasets) == 5
    _assert_datasets_have_id_and_valid_status(datasets)


def test_list_datasets_by_number_instances(all_datasets: pd.DataFrame):
    small_datasets = openml.datasets.list_datasets(
        number_instances="5..100",
        output_format="dataframe",
    )
    assert 0 < len(small_datasets) <= len(all_datasets)
    _assert_datasets_have_id_and_valid_status(small_datasets)


def test_list_datasets_by_number_features(all_datasets: pd.DataFrame):
    wide_datasets = openml.datasets.list_datasets(
        number_features="50..100",
        output_format="dataframe",
    )
    assert 8 <= len(wide_datasets) < len(all_datasets)
    _assert_datasets_have_id_and_valid_status(wide_datasets)


def test_list_datasets_by_number_classes(all_datasets: pd.DataFrame):
    five_class_datasets = openml.datasets.list_datasets(
        number_classes="5",
        output_format="dataframe",
    )
    assert 3 <= len(five_class_datasets) < len(all_datasets)
    _assert_datasets_have_id_and_valid_status(five_class_datasets)


def test_list_datasets_by_number_missing_values(all_datasets: pd.DataFrame):
    na_datasets = openml.datasets.list_datasets(
        number_missing_values="5..100",
        output_format="dataframe",
    )
    assert 5 <= len(na_datasets) < len(all_datasets)
    _assert_datasets_have_id_and_valid_status(na_datasets)


def test_list_datasets_combined_filters(all_datasets: pd.DataFrame):
    combined_filter_datasets = openml.datasets.list_datasets(
        tag="study_14",
        number_instances="100..1000",
        number_missing_values="800..1000",
        output_format="dataframe",
    )
    assert 1 <= len(combined_filter_datasets) < len(all_datasets)
    _assert_datasets_have_id_and_valid_status(combined_filter_datasets)
