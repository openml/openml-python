# License: BSD 3-Clause
from __future__ import annotations

import argparse
from unittest import mock

import pandas as pd
import pytest

from openml import cli


class TestDatasetsCLI:
    """Test suite for datasets CLI commands."""

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_list_basic(self, mock_list):
        """Test basic dataset listing."""
        # Mock return value
        mock_df = pd.DataFrame({
            "did": [1, 2, 3],
            "name": ["iris", "wine", "anneal"],
            "status": ["active", "active", "active"],
        })
        mock_list.return_value = mock_df

        # Create args
        args = argparse.Namespace(
            offset=None,
            size=10,
            tag=None,
            status=None,
            data_name=None,
            number_instances=None,
            number_features=None,
            number_classes=None,
            format="table",
            verbose=False,
        )

        # Execute
        cli.datasets_list(args)

        # Verify
        mock_list.assert_called_once_with(offset=None, size=10)

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_list_with_filters(self, mock_list):
        """Test dataset listing with filters."""
        mock_df = pd.DataFrame({"did": [1], "name": ["iris"]})
        mock_list.return_value = mock_df

        args = argparse.Namespace(
            offset=0,
            size=5,
            tag="study_14",
            status="active",
            data_name=None,
            number_instances="100..1000",
            number_features=None,
            number_classes=None,
            format="table",
            verbose=False,
        )

        cli.datasets_list(args)

        mock_list.assert_called_once_with(
            offset=0,
            size=5,
            tag="study_14",
            status="active",
            number_instances="100..1000",
        )

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_list_empty_results(self, mock_list):
        """Test handling of empty results."""
        mock_list.return_value = pd.DataFrame()

        args = argparse.Namespace(
            offset=None,
            size=None,
            tag=None,
            status=None,
            data_name=None,
            number_instances=None,
            number_features=None,
            number_classes=None,
            format="table",
            verbose=False,
        )

        cli.datasets_list(args)
        mock_list.assert_called_once()

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_list_json_format(self, mock_list):
        """Test JSON output format."""
        mock_df = pd.DataFrame({"did": [1], "name": ["iris"]})
        mock_list.return_value = mock_df

        args = argparse.Namespace(
            offset=None,
            size=10,
            tag=None,
            status=None,
            data_name=None,
            number_instances=None,
            number_features=None,
            number_classes=None,
            format="json",
            verbose=False,
        )

        cli.datasets_list(args)
        mock_list.assert_called_once()

    @mock.patch("openml.datasets.get_dataset")
    def test_datasets_info(self, mock_get):
        """Test dataset info display."""
        # Create mock dataset
        mock_dataset = mock.Mock()
        mock_dataset.dataset_id = 61
        mock_dataset.name = "iris"
        mock_dataset.version = 1
        mock_dataset.status = "active"
        mock_dataset.format = "ARFF"
        mock_dataset.upload_date = "2014-04-06 23:19:17"
        mock_dataset.description = "Famous iris dataset"
        mock_dataset.qualities = {
            "NumberOfInstances": 150,
            "NumberOfFeatures": 5,
            "NumberOfClasses": 3,
        }
        mock_dataset.features = {"sepallength": mock.Mock(data_type="numeric")}
        mock_get.return_value = mock_dataset

        args = argparse.Namespace(dataset_id="61")

        cli.datasets_info(args)

        mock_get.assert_called_once_with(
            "61",
            download_data=False,
            download_qualities=True,
            download_features_meta_data=True,
        )

    @mock.patch("openml.datasets.get_dataset")
    def test_datasets_info_error(self, mock_get):
        """Test dataset info with invalid ID."""
        mock_get.side_effect = Exception("Dataset not found")

        args = argparse.Namespace(dataset_id="99999")

        with pytest.raises(SystemExit):
            cli.datasets_info(args)

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_search_found(self, mock_list):
        """Test search with results."""
        mock_df = pd.DataFrame({"did": [61], "name": ["iris"]})
        mock_list.return_value = mock_df

        args = argparse.Namespace(query="iris", size=20, format="table", verbose=False)

        cli.datasets_search(args)

        # Should try exact match first
        mock_list.assert_called()

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_search_not_found(self, mock_list):
        """Test search with no results."""
        mock_list.return_value = pd.DataFrame()

        args = argparse.Namespace(query="nonexistent", size=20, format="table", verbose=False)

        cli.datasets_search(args)
        assert mock_list.call_count >= 1

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_search_case_insensitive(self, mock_list):
        """Test case-insensitive search."""
        # First call returns empty (no exact match)
        # Second call returns all datasets for client-side filtering
        mock_list.side_effect = [
            pd.DataFrame(),  # No exact match
            pd.DataFrame({"did": [61, 62], "name": ["Iris", "IRIS-versicolor"]}),
        ]

        args = argparse.Namespace(query="iris", size=20, format="table", verbose=False)

        cli.datasets_search(args)
        assert mock_list.call_count == 2

    def test_datasets_handler_no_action(self):
        """Test datasets handler with no action specified."""
        args = argparse.Namespace(datasets_action=None)

        with pytest.raises(SystemExit):
            cli.datasets_handler(args)

    @mock.patch("openml.datasets.list_datasets")
    def test_datasets_handler_list_action(self, mock_list):
        """Test datasets handler routes list action correctly."""
        mock_list.return_value = pd.DataFrame({"did": [1], "name": ["test"]})

        args = argparse.Namespace(
            datasets_action="list",
            offset=None,
            size=5,
            tag=None,
            status=None,
            data_name=None,
            number_instances=None,
            number_features=None,
            number_classes=None,
            format="table",
            verbose=False,
        )

        cli.datasets_handler(args)
        mock_list.assert_called_once()
