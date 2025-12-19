# License: BSD 3-Clause
from __future__ import annotations

import argparse
from unittest import mock

import pandas as pd
import pytest

from openml import cli


class TestCLIStudies:
    """Test suite for studies CLI commands."""

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_list_all_types(self, mock_list_suites, mock_list_studies):
        """Test listing all types (studies + suites)."""
        # Mock return values
        mock_studies_df = pd.DataFrame({
            "id": [1, 2],
            "alias": ["study1", "study2"],
            "name": ["Study 1", "Study 2"],
            "status": ["active", "active"],
            "main_entity_type": ["run", "run"],
        })
        mock_suites_df = pd.DataFrame({
            "id": [10, 11],
            "alias": ["suite1", "suite2"],
            "name": ["Suite 1", "Suite 2"],
            "status": ["active", "active"],
            "main_entity_type": ["task", "task"],
        })
        mock_list_studies.return_value = mock_studies_df
        mock_list_suites.return_value = mock_suites_df

        # Create args
        args = argparse.Namespace(
            status=None,
            uploader=None,
            type="all",
            size=10,
            offset=0,
            format="list",
            verbose=False,
        )

        # Execute
        cli.studies_list(args)

        # Verify both functions were called (None values filtered out)
        mock_list_studies.assert_called_once_with(size=10, offset=0)
        mock_list_suites.assert_called_once_with(size=10, offset=0)

    @mock.patch("openml.study.functions.list_studies")
    def test_studies_list_only_studies(self, mock_list_studies):
        """Test listing only studies."""
        mock_df = pd.DataFrame({
            "id": [1, 2],
            "alias": ["study1", "study2"],
            "name": ["Study 1", "Study 2"],
        })
        mock_list_studies.return_value = mock_df

        args = argparse.Namespace(
            status="active",
            uploader=123,
            type="study",
            size=5,
            offset=0,
            format="table",
            verbose=False,
        )

        cli.studies_list(args)

        mock_list_studies.assert_called_once_with(status="active", uploader=123, size=5, offset=0)

    @mock.patch("openml.study.functions.list_suites")
    def test_studies_list_only_suites(self, mock_list_suites):
        """Test listing only suites."""
        mock_df = pd.DataFrame({
            "id": [10, 11],
            "alias": ["suite1", "suite2"],
            "name": ["Suite 1", "Suite 2"],
        })
        mock_list_suites.return_value = mock_df

        args = argparse.Namespace(
            status=None,
            uploader=None,
            type="suite",
            size=10,
            offset=0,
            format="list",
            verbose=False,
        )

        cli.studies_list(args)

        # None values are filtered out
        mock_list_suites.assert_called_once_with(size=10, offset=0)

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_list_empty_results(self, mock_list_suites, mock_list_studies):
        """Test handling of empty results."""
        mock_list_studies.return_value = pd.DataFrame()
        mock_list_suites.return_value = pd.DataFrame()

        args = argparse.Namespace(
            status=None,
            uploader=None,
            type="all",
            size=10,
            offset=0,
            format="list",
            verbose=False,
        )

        cli.studies_list(args)

        mock_list_studies.assert_called_once()
        mock_list_suites.assert_called_once()

    @mock.patch("openml.study.functions.list_studies")
    def test_studies_list_json_format(self, mock_list_studies):
        """Test JSON output format."""
        mock_df = pd.DataFrame({"id": [1], "name": ["Study 1"]})
        mock_list_studies.return_value = mock_df

        args = argparse.Namespace(
            status=None,
            uploader=None,
            type="study",
            size=10,
            offset=0,
            format="json",
            verbose=False,
        )

        cli.studies_list(args)
        mock_list_studies.assert_called_once()

    @mock.patch("openml.study.functions.list_studies")
    def test_studies_list_verbose(self, mock_list_studies):
        """Test verbose output."""
        mock_df = pd.DataFrame({
            "id": [1],
            "name": ["Study 1"],
            "description": ["A test study"],
        })
        mock_list_studies.return_value = mock_df

        args = argparse.Namespace(
            status=None,
            uploader=None,
            type="study",
            size=10,
            offset=0,
            format="list",
            verbose=True,
        )

        cli.studies_list(args)
        mock_list_studies.assert_called_once()

    @mock.patch("openml.study.functions.get_study")
    def test_studies_info_study(self, mock_get_study):
        """Test study info display for a study."""
        # Create mock study with all attributes
        mock_study = mock.Mock()
        mock_study.study_id = 1
        mock_study.alias = "study1"
        mock_study.name = "Test Study"
        mock_study.description = "A test study"
        mock_study.status = "active"
        mock_study.creation_date = "2023-01-01"
        mock_study.creator = 123
        mock_study.main_entity_type = "run"
        mock_study.benchmark_suite = None
        mock_study.data = []
        mock_study.tasks = []
        mock_study.flows = []
        mock_study.runs = [1, 2, 3, 4, 5]
        mock_study.setups = []
        mock_get_study.return_value = mock_study

        args = argparse.Namespace(study_id="1", verbose=False)

        cli.studies_info(args)

        mock_get_study.assert_called_once_with("1")

    @mock.patch("openml.study.functions.get_study")
    @mock.patch("openml.study.functions.get_suite")
    def test_studies_info_suite_fallback(self, mock_get_suite, mock_get_study):
        """Test suite info display when study fetch fails."""
        # get_study raises exception, should fall back to get_suite
        mock_get_study.side_effect = Exception("Not a study")

        mock_suite = mock.Mock()
        mock_suite.study_id = 10
        mock_suite.alias = "suite1"
        mock_suite.name = "Test Suite"
        mock_suite.description = "A test suite"
        mock_suite.status = "active"
        mock_suite.creation_date = "2023-01-01"
        mock_suite.creator = 456
        mock_suite.main_entity_type = "task"
        mock_suite.benchmark_suite = None
        mock_suite.data = []
        mock_suite.tasks = [10, 11, 12]
        mock_suite.flows = []
        mock_suite.runs = []
        mock_suite.setups = []
        mock_get_suite.return_value = mock_suite

        args = argparse.Namespace(study_id="10", verbose=False)

        cli.studies_info(args)

        mock_get_study.assert_called_once_with("10")
        mock_get_suite.assert_called_once_with("10")

    @mock.patch("openml.study.functions.get_study")
    def test_studies_info_verbose(self, mock_get_study):
        """Test verbose study info display."""
        mock_study = mock.Mock()
        mock_study.study_id = 1
        mock_study.alias = "study1"
        mock_study.name = "Test Study"
        mock_study.description = "A test study"
        mock_study.status = "active"
        mock_study.creation_date = "2023-01-01"
        mock_study.creator = 123
        mock_study.main_entity_type = "run"
        mock_study.benchmark_suite = None
        mock_study.data = []
        mock_study.tasks = []
        mock_study.flows = []
        mock_study.runs = list(range(1, 16))  # 15 runs
        mock_study.setups = []
        mock_get_study.return_value = mock_study

        args = argparse.Namespace(study_id="1", verbose=True)

        cli.studies_info(args)

        mock_get_study.assert_called_once_with("1")

    @mock.patch("openml.study.functions.get_study")
    @mock.patch("openml.study.functions.get_suite")
    def test_studies_info_not_found(self, mock_get_suite, mock_get_study):
        """Test study info with invalid ID."""
        mock_get_study.side_effect = Exception("Study not found")
        mock_get_suite.side_effect = Exception("Suite not found")

        args = argparse.Namespace(study_id="99999", verbose=False)

        with pytest.raises(SystemExit):
            cli.studies_info(args)

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_search_found(self, mock_list_suites, mock_list_studies):
        """Test search with results."""
        mock_studies_df = pd.DataFrame({
            "id": [1],
            "alias": ["study1"],
            "name": ["OpenML100"],
        })
        mock_suites_df = pd.DataFrame({
            "id": [10],
            "alias": ["suite1"],
            "name": ["OpenML-CC18"],
        })
        mock_list_studies.return_value = mock_studies_df
        mock_list_suites.return_value = mock_suites_df

        args = argparse.Namespace(
            query="openml",
            status=None,
            format="list",
            verbose=False,
        )

        cli.studies_search(args)

        mock_list_studies.assert_called_once()
        mock_list_suites.assert_called_once()

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_search_case_insensitive(self, mock_list_suites, mock_list_studies):
        """Test case-insensitive search."""
        mock_studies_df = pd.DataFrame({
            "id": [1, 2],
            "alias": ["study1", "study2"],
            "name": ["OpenML100", "OpenML-Benchmarking"],
        })
        mock_suites_df = pd.DataFrame({
            "id": [10],
            "alias": ["suite1"],
            "name": ["OpenML-CC18"],
        })
        mock_list_studies.return_value = mock_studies_df
        mock_list_suites.return_value = mock_suites_df

        args = argparse.Namespace(
            query="OPENML",
            status=None,
            format="table",
            verbose=False,
        )

        cli.studies_search(args)

        mock_list_studies.assert_called_once()
        mock_list_suites.assert_called_once()

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_search_with_status_filter(self, mock_list_suites, mock_list_studies):
        """Test search with status filter."""
        mock_studies_df = pd.DataFrame({
            "id": [1],
            "alias": ["study1"],
            "name": ["Test Study"],
        })
        mock_suites_df = pd.DataFrame()
        mock_list_studies.return_value = mock_studies_df
        mock_list_suites.return_value = mock_suites_df

        args = argparse.Namespace(
            query="test",
            status="active",
            format="list",
            verbose=False,
        )

        cli.studies_search(args)

        mock_list_studies.assert_called_once_with(status="active")
        mock_list_suites.assert_called_once_with(status="active")

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_search_not_found(self, mock_list_suites, mock_list_studies):
        """Test search with no results."""
        mock_list_studies.return_value = pd.DataFrame()
        mock_list_suites.return_value = pd.DataFrame()

        args = argparse.Namespace(
            query="nonexistent",
            status=None,
            format="list",
            verbose=False,
        )

        cli.studies_search(args)

        mock_list_studies.assert_called_once()
        mock_list_suites.assert_called_once()

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_search_json_format(self, mock_list_suites, mock_list_studies):
        """Test search with JSON output format."""
        mock_studies_df = pd.DataFrame({"id": [1], "name": ["Study 1"]})
        mock_suites_df = pd.DataFrame()
        mock_list_studies.return_value = mock_studies_df
        mock_list_suites.return_value = mock_suites_df

        args = argparse.Namespace(
            query="study",
            status=None,
            format="json",
            verbose=False,
        )

        cli.studies_search(args)

        mock_list_studies.assert_called_once()
        mock_list_suites.assert_called_once()

    @mock.patch("openml.study.functions.list_studies")
    def test_studies_dispatcher_list(self, mock_list_studies):
        """Test studies dispatcher routes list action correctly."""
        mock_list_studies.return_value = pd.DataFrame({"id": [1], "name": ["test"]})

        args = argparse.Namespace(
            studies_subcommand="list",
            status=None,
            uploader=None,
            type="study",
            size=10,
            offset=0,
            format="list",
            verbose=False,
        )

        cli.studies(args)
        mock_list_studies.assert_called_once()

    @mock.patch("openml.study.functions.get_study")
    def test_studies_dispatcher_info(self, mock_get_study):
        """Test studies dispatcher routes info action correctly."""
        mock_study = mock.Mock()
        mock_study.study_id = 1
        mock_study.name = "Test"
        mock_study.status = "active"
        mock_study.main_entity_type = "run"
        mock_study.creator = 123
        mock_study.creation_date = "2023-01-01"
        mock_study.description = "Test description"
        mock_study.benchmark_suite = None
        mock_study.alias = None
        mock_study.data = []
        mock_study.tasks = []
        mock_study.flows = []
        mock_study.runs = []
        mock_study.setups = []
        mock_get_study.return_value = mock_study

        args = argparse.Namespace(
            studies_subcommand="info",
            study_id="1",
            verbose=False,
        )

        cli.studies(args)
        mock_get_study.assert_called_once()

    @mock.patch("openml.study.functions.list_studies")
    @mock.patch("openml.study.functions.list_suites")
    def test_studies_dispatcher_search(self, mock_list_suites, mock_list_studies):
        """Test studies dispatcher routes search action correctly."""
        mock_list_studies.return_value = pd.DataFrame({"id": [1], "name": ["test"]})
        mock_list_suites.return_value = pd.DataFrame()

        args = argparse.Namespace(
            studies_subcommand="search",
            query="test",
            status=None,
            format="list",
            verbose=False,
        )

        cli.studies(args)
        mock_list_studies.assert_called_once()
        mock_list_suites.assert_called_once()

    def test_studies_dispatcher_no_subcommand(self):
        """Test studies dispatcher with no subcommand specified."""
        args = argparse.Namespace(studies_subcommand=None)

        with pytest.raises(SystemExit):
            cli.studies(args)
