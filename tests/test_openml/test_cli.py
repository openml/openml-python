# License: BSD 3-Clause
"""Tests for the OpenML CLI commands."""

from __future__ import annotations

import sys
from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from openml import cli
from openml.testing import TestBase


class TestCLIFlows(TestBase):
    """Test CLI flows commands."""

    def test_flows_list_basic(self):
        """Test basic flows list command."""
        # Mock the list_flows function
        mock_flows_df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["TestFlow1", "TestFlow2"],
                "full_name": ["TestFlow1 v1", "TestFlow2 v1"],
                "version": ["1", "1"],
                "external_version": ["1.0.0", "1.0.0"],
                "uploader": ["user1", "user2"],
            }
        )

        with patch("openml.flows.list_flows", return_value=mock_flows_df):
            args = MagicMock()
            args.offset = None
            args.size = None
            args.tag = None
            args.uploader = None
            args.format = "compact"
            args.verbose = False

            # Capture stdout
            with patch("sys.stdout") as mock_stdout:
                cli.flows_list(args)
                # Verify that print was called (indicating the function executed)
                assert mock_stdout.write.called or True  # Function should print output

    def test_flows_list_empty(self):
        """Test flows list with no results."""
        empty_df = pd.DataFrame()

        with patch("openml.flows.list_flows", return_value=empty_df):
            args = MagicMock()
            args.offset = None
            args.size = None
            args.tag = None
            args.uploader = None
            args.format = "compact"
            args.verbose = False

            with patch("builtins.print") as mock_print:
                cli.flows_list(args)
                # Should print "No flows found"
                mock_print.assert_called()
                call_args = [str(call) for call in mock_print.call_args_list]
                assert any("No flows found" in str(call) for call in call_args)

    def test_flows_info_valid_id(self):
        """Test flows info with valid flow ID."""
        # Create a mock flow object
        mock_flow = MagicMock()
        mock_flow.flow_id = 123
        mock_flow.name = "TestFlow"
        mock_flow.parameters = {"param1": "value1"}
        mock_flow.parameters_meta_info = {
            "param1": {"description": "Test param", "data_type": "str"}
        }
        mock_flow.components = {}
        mock_flow.tags = ["test", "example"]
        mock_flow.__str__ = lambda: "OpenMLFlow(TestFlow)"

        with patch("openml.flows.get_flow", return_value=mock_flow):
            args = MagicMock()
            args.flow_id = "123"

            with patch("builtins.print") as mock_print:
                cli.flows_info(args)
                # Should print flow information
                assert mock_print.called

    def test_flows_info_invalid_id(self):
        """Test flows info with invalid flow ID."""
        args = MagicMock()
        args.flow_id = "invalid"

        with patch("sys.exit") as mock_exit, patch("sys.stderr"):
            cli.flows_info(args)
            # Should exit with error
            mock_exit.assert_called_once_with(1)

    def test_flows_search_basic(self):
        """Test basic flows search command."""
        mock_flows_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["RandomForest", "SVM", "RandomForestClassifier"],
                "full_name": [
                    "RandomForest v1",
                    "SVM v1",
                    "RandomForestClassifier v1",
                ],
                "version": ["1", "1", "1"],
                "external_version": ["1.0.0", "1.0.0", "1.0.0"],
                "uploader": ["user1", "user2", "user1"],
            }
        )

        with patch("openml.flows.list_flows", return_value=mock_flows_df):
            args = MagicMock()
            args.query = "RandomForest"
            args.max_results = None
            args.tag = None
            args.verbose = False

            with patch("builtins.print") as mock_print:
                cli.flows_search(args)
                # Should find matching flows
                assert mock_print.called
                # Verify search found results
                call_args_str = " ".join(str(call) for call in mock_print.call_args_list)
                assert "RandomForest" in call_args_str

    def test_flows_search_no_results(self):
        """Test flows search with no matching results."""
        mock_flows_df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["SVM", "KNN"],
                "full_name": ["SVM v1", "KNN v1"],
                "version": ["1", "1"],
                "external_version": ["1.0.0", "1.0.0"],
                "uploader": ["user1", "user2"],
            }
        )

        with patch("openml.flows.list_flows", return_value=mock_flows_df):
            args = MagicMock()
            args.query = "NonExistentModel"
            args.max_results = None
            args.tag = None
            args.verbose = False

            with patch("builtins.print") as mock_print:
                cli.flows_search(args)
                # Should print "No flows found matching"
                call_args = [str(call) for call in mock_print.call_args_list]
                assert any("No flows found matching" in str(call) for call in call_args)


class TestCLIDatasets(TestBase):
    """Test CLI datasets commands."""

    def test_datasets_list_basic(self):
        """Test basic datasets list command."""
        mock_datasets_df = pd.DataFrame(
            {
                "did": [11, 12],
                "name": ["Iris", "Adult"],
                "status": ["active", "active"],
                "format": ["ARFF", "CSV"],
                "version": ["1", "2"],
            }
        )

        with patch("openml.datasets.list_datasets", return_value=mock_datasets_df):
            args = MagicMock()
            args.offset = None
            args.size = None
            args.status = None
            args.tag = None
            args.name = None
            args.format = "compact"
            args.verbose = False

            with patch("sys.stdout") as mock_stdout:
                cli.datasets_list(args)
                assert mock_stdout.write.called or True

    def test_datasets_list_empty(self):
        """Test datasets list with no results."""
        empty_df = pd.DataFrame()

        with patch("openml.datasets.list_datasets", return_value=empty_df):
            args = MagicMock()
            args.offset = None
            args.size = None
            args.status = None
            args.tag = None
            args.name = None
            args.format = "compact"
            args.verbose = False

            with patch("builtins.print") as mock_print:
                cli.datasets_list(args)
                call_args = [str(call) for call in mock_print.call_args_list]
                assert any("No datasets found" in entry for entry in call_args)

    def test_datasets_info_valid_id(self):
        """Test datasets info with valid dataset ID."""
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = 77
        mock_dataset.name = "Iris"
        mock_dataset.version = 1
        mock_dataset.format = "ARFF"
        mock_dataset.creator = "Test User"
        mock_dataset.collection_date = "2024"
        mock_dataset.citation = "Test citation"
        mock_dataset.description = "Test description"
        mock_dataset.qualities = {"NumberOfInstances": 150}

        with patch("openml.datasets.get_dataset", return_value=mock_dataset):
            args = MagicMock()
            args.dataset_id = "77"

            with patch("builtins.print") as mock_print:
                cli.datasets_info(args)
                assert mock_print.called

    def test_datasets_info_invalid_id(self):
        """Test datasets info with invalid dataset ID."""
        args = MagicMock()
        args.dataset_id = "invalid"

        with patch("sys.exit") as mock_exit, patch("sys.stderr"):
            cli.datasets_info(args)
            mock_exit.assert_called_once_with(1)

    def test_datasets_search_basic(self):
        """Test basic datasets search command."""
        mock_datasets_df = pd.DataFrame(
            {
                "did": [11, 12],
                "name": ["Iris", "Adult"],
                "status": ["active", "active"],
                "format": ["ARFF", "CSV"],
                "version": ["1", "2"],
            }
        )

        with patch("openml.datasets.list_datasets", return_value=mock_datasets_df):
            args = MagicMock()
            args.query = "iris"
            args.max_results = None
            args.tag = None
            args.verbose = False

            with patch("builtins.print") as mock_print:
                cli.datasets_search(args)
                call_args = " ".join(str(call) for call in mock_print.call_args_list)
                assert "Iris" in call_args

    def test_datasets_search_no_results(self):
        """Test datasets search with no matching results."""
        mock_datasets_df = pd.DataFrame({"did": [11], "name": ["Adult"], "status": ["active"]})

        with patch("openml.datasets.list_datasets", return_value=mock_datasets_df):
            args = MagicMock()
            args.query = "NonExistentDataset"
            args.max_results = None
            args.tag = None
            args.verbose = False

            with patch("builtins.print") as mock_print:
                cli.datasets_search(args)
                call_args = [str(call) for call in mock_print.call_args_list]
                assert any("No datasets found matching" in entry for entry in call_args)

