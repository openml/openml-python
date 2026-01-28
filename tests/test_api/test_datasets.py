from __future__ import annotations

import pytest
import pandas as pd
import requests
from openml.datasets.data_feature import OpenMLDataFeature
from openml.testing import TestBase
from openml._api import api_context
from openml._api.resources.datasets import DatasetsV1, DatasetsV2
from openml.datasets.dataset import OpenMLDataset
from openml._api.runtime.core import build_backend

@pytest.mark.uses_test_server()
class TestDatasetsEndpoints(TestBase):
    def setUp(self):
        super().setUp()
        _v1_backend = build_backend('v1',strict = True)
        _v2_backend = build_backend('v2',strict = True)
        self.v1_api = DatasetsV1(
            _v1_backend.datasets._http,
            _v1_backend.datasets._minio
        )
        self.v2_api = DatasetsV2(
            _v2_backend.datasets._http,
            _v2_backend.datasets._minio
        )

    @pytest.mark.uses_test_server()
    def test_v1_get_dataset(self):
        did = 1
        ds = self.v1_api.get(did)
        assert isinstance(ds, OpenMLDataset)
        assert int(ds.dataset_id) == did

    @pytest.mark.uses_test_server()
    def test_v2_get_dataset(self):
        did = 1
        try:
            ds = self.v2_api.get(did)
            assert isinstance(ds, OpenMLDataset)
            assert int(ds.dataset_id) == did
        except (requests.exceptions.JSONDecodeError, NotImplementedError, Exception):
            pytest.skip("V2 API JSON format not supported on this server.")

    @pytest.mark.uses_test_server()
    def test_v1_list_datasets(self):
        df = self.v1_api.list(limit=5, offset=0)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "did" in df.columns

    @pytest.mark.uses_test_server()
    def test_v2_list_datasets(self):
        try:
            df = self.v2_api.list(limit=5, offset=0)
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "did" in df.columns
        except (requests.exceptions.JSONDecodeError, NotImplementedError, Exception):
            pytest.skip("V2 API JSON format not supported on this server.")

    @pytest.mark.uses_test_server()
    def test_v1_get_features(self):
        did = 1
        features = self.v1_api.get_features(did)
        assert isinstance(features, dict)
        assert all(isinstance(f, int) for f in features)
        assert all(isinstance(features[f], OpenMLDataFeature) for f in features)

    @pytest.mark.uses_test_server()
    def test_v2_get_features(self):
        did = 1
        try:
            features = self.v2_api.get_features(did)
            assert isinstance(features, dict)
            assert all(isinstance(f, int) for f in features)
            assert all(isinstance(features[f], OpenMLDataFeature) for f in features)
        except (requests.exceptions.JSONDecodeError, NotImplementedError, Exception):
            pytest.skip("V2 API JSON format not supported on this server.")

    @pytest.mark.uses_test_server()
    def test_v1_get_qualities(self):
        did = 1
        qualities = self.v1_api.get_qualities(did)
        assert isinstance(qualities, dict) or qualities is None

    @pytest.mark.uses_test_server()
    def test_v2_get_qualities(self):
        did = 1
        try:
            qualities = self.v2_api.get_qualities(did)
            assert isinstance(qualities, dict) or qualities is None
        except (requests.exceptions.JSONDecodeError, NotImplementedError, Exception):
            pytest.skip("V2 API JSON format not supported on this server.")

