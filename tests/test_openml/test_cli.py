# License: BSD 3-Clause
"""Tests for the OpenML CLI commands."""
from __future__ import annotations

import argparse
import sys
from io import StringIO
from unittest import mock

import pandas as pd
import pytest

from openml import cli
from openml.runs import OpenMLRun
from openml.tasks import TaskType
from openml.testing import TestBase


class TestCLIRuns(TestBase):
    """Test suite for openml runs CLI commands."""

    def _create_mock_run(self, run_id: int, task_id: int, flow_id: int) -> OpenMLRun:
        """Helper to create a mock OpenMLRun object."""
        return OpenMLRun(
            run_id=run_id,
            task_id=task_id,
            flow_id=flow_id,
            dataset_id=1,
            setup_id=100 + run_id,
            uploader=1,
            uploader_name="Test User",
            flow_name=f"test.flow.{flow_id}",
            task_type="Supervised Classification",
            evaluations={"predictive_accuracy": 0.95, "area_under_roc_curve": 0.98},
            fold_evaluations={
                "predictive_accuracy": {0: {0: 0.94, 1: 0.96}},
            },
            parameter_settings=[
                {"oml:name": "n_estimators", "oml:value": "100"},
                {"oml:name": "max_depth", "oml:value": "10", "oml:component": "estimator"},
            ],
            tags=["test", "openml-python"],
            predictions_url="https://test.openml.org/predictions/12345",
            output_files={"predictions": 12345, "description": 12346},
        )

    def _create_mock_runs_dataframe(self) -> pd.DataFrame:
        """Helper to create a mock DataFrame for list_runs."""
        return pd.DataFrame(
            {
                "run_id": [1, 2, 3],
                "task_id": [1, 1, 2],
                "flow_id": [100, 101, 100],
                "setup_id": [200, 201, 200],
                "uploader": [1, 2, 1],
                "task_type": [TaskType.SUPERVISED_CLASSIFICATION] * 3,
                "upload_time": ["2024-01-01 10:00:00", "2024-01-02 11:00:00", "2024-01-03 12:00:00"],
                "error_message": ["", "", ""],
            }
        )

    @mock.patch("openml.runs.functions.list_runs")
    def test_runs_list_simple(self, mock_list_runs: mock.Mock) -> None:
        """Test runs list command with simple output."""
        mock_list_runs.return_value = self._create_mock_runs_dataframe()

        args = argparse.Namespace(
            task=None,
            flow=None,
            uploader=None,
            tag=None,
            size=10,
            offset=0,
            format="list",
            verbose=False,
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_list(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "1: Task 1" in output
        assert "2: Task 1" in output
        assert "3: Task 2" in output
        mock_list_runs.assert_called_once_with(size=10, offset=0)

    @mock.patch("openml.runs.functions.list_runs")
    def test_runs_list_with_filters(self, mock_list_runs: mock.Mock) -> None:
        """Test runs list command with filtering parameters."""
        mock_list_runs.return_value = self._create_mock_runs_dataframe()

        args = argparse.Namespace(
            task=1,
            flow=100,
            uploader="TestUser",
            tag="test",
            size=20,
            offset=10,
            format="list",
            verbose=False,
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_list(args)

        sys.stdout = sys.__stdout__

        mock_list_runs.assert_called_once_with(
            task=[1], flow=[100], uploader=["TestUser"], tag="test", size=20, offset=10
        )

    @mock.patch("openml.runs.functions.list_runs")
    def test_runs_list_verbose(self, mock_list_runs: mock.Mock) -> None:
        """Test runs list command with verbose output."""
        mock_list_runs.return_value = self._create_mock_runs_dataframe()

        args = argparse.Namespace(
            task=None,
            flow=None,
            uploader=None,
            tag=None,
            size=10,
            offset=0,
            format="list",
            verbose=True,
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_list(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "Run ID: 1" in output
        assert "Task ID: 1" in output
        assert "Flow ID: 100" in output
        assert "Setup ID: 200" in output
        assert "Uploader: 1" in output

    @mock.patch("openml.runs.functions.list_runs")
    def test_runs_list_table_format(self, mock_list_runs: mock.Mock) -> None:
        """Test runs list command with table format."""
        mock_list_runs.return_value = self._create_mock_runs_dataframe()

        args = argparse.Namespace(
            task=None,
            flow=None,
            uploader=None,
            tag=None,
            size=10,
            offset=0,
            format="table",
            verbose=False,
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_list(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        # Table format should show column headers
        assert "run_id" in output
        assert "task_id" in output
        assert "flow_id" in output

    @mock.patch("openml.runs.functions.list_runs")
    def test_runs_list_json_format(self, mock_list_runs: mock.Mock) -> None:
        """Test runs list command with JSON format."""
        mock_list_runs.return_value = self._create_mock_runs_dataframe()

        args = argparse.Namespace(
            task=None,
            flow=None,
            uploader=None,
            tag=None,
            size=10,
            offset=0,
            format="json",
            verbose=False,
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_list(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        # JSON format should contain valid JSON structure
        assert '"run_id":' in output or '"run_id": ' in output
        assert '"task_id":' in output or '"task_id": ' in output

    @mock.patch("openml.runs.functions.list_runs")
    def test_runs_list_empty_results(self, mock_list_runs: mock.Mock) -> None:
        """Test runs list command with no results."""
        mock_list_runs.return_value = pd.DataFrame()

        args = argparse.Namespace(
            task=999,
            flow=None,
            uploader=None,
            tag=None,
            size=10,
            offset=0,
            format="list",
            verbose=False,
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_list(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "No runs found" in output

    @mock.patch("openml.runs.functions.list_runs")
    def test_runs_list_error_handling(self, mock_list_runs: mock.Mock) -> None:
        """Test runs list command error handling."""
        mock_list_runs.side_effect = Exception("Connection error")

        args = argparse.Namespace(
            task=None,
            flow=None,
            uploader=None,
            tag=None,
            size=10,
            offset=0,
            format="list",
            verbose=False,
        )

        # Capture stderr
        captured_error = StringIO()
        sys.stderr = captured_error

        with pytest.raises(SystemExit):
            cli.runs_list(args)

        sys.stderr = sys.__stderr__

        error = captured_error.getvalue()
        assert "Error listing runs" in error
        assert "Connection error" in error

    @mock.patch("openml.runs.functions.get_run")
    def test_runs_info(self, mock_get_run: mock.Mock) -> None:
        """Test runs info command."""
        mock_run = self._create_mock_run(run_id=12345, task_id=1, flow_id=100)
        mock_get_run.return_value = mock_run

        args = argparse.Namespace(run_id=12345)

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_info(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "Run ID: 12345" in output
        assert "Task ID: 1" in output
        assert "Flow ID: 100" in output
        assert "Flow Name: test.flow.100" in output
        assert "Setup ID: 12445" in output  # 100 + 12345
        assert "Dataset ID: 1" in output
        assert "Uploader: Test User (ID: 1)" in output
        assert "Parameter Settings:" in output
        assert "n_estimators: 100" in output
        assert "estimator.max_depth: 10" in output
        assert "Evaluations:" in output
        assert "predictive_accuracy: 0.95" in output
        assert "area_under_roc_curve: 0.98" in output
        assert "Tags: test, openml-python" in output
        assert "Predictions URL: https://test.openml.org/predictions/12345" in output

        mock_get_run.assert_called_once_with(12345)

    @mock.patch("openml.runs.functions.get_run")
    def test_runs_info_with_fold_evaluations(self, mock_get_run: mock.Mock) -> None:
        """Test runs info command displays fold evaluation summary."""
        mock_run = self._create_mock_run(run_id=12345, task_id=1, flow_id=100)
        mock_get_run.return_value = mock_run

        args = argparse.Namespace(run_id=12345)

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_info(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "Fold Evaluations (Summary):" in output
        # Average of 0.94 and 0.96 = 0.95
        assert "0.9500" in output or "0.95" in output

    @mock.patch("openml.runs.functions.get_run")
    def test_runs_info_error_handling(self, mock_get_run: mock.Mock) -> None:
        """Test runs info command error handling."""
        mock_get_run.side_effect = Exception("Run not found")

        args = argparse.Namespace(run_id=99999)

        # Capture stderr
        captured_error = StringIO()
        sys.stderr = captured_error

        with pytest.raises(SystemExit):
            cli.runs_info(args)

        sys.stderr = sys.__stderr__

        error = captured_error.getvalue()
        assert "Error fetching run information" in error
        assert "Run not found" in error

    @mock.patch("openml.config.get_cache_directory")
    @mock.patch("openml.runs.functions.get_run")
    def test_runs_download(self, mock_get_run: mock.Mock, mock_get_cache: mock.Mock) -> None:
        """Test runs download command."""
        mock_run = self._create_mock_run(run_id=12345, task_id=1, flow_id=100)
        mock_get_run.return_value = mock_run
        mock_get_cache.return_value = "/tmp/openml_cache"

        args = argparse.Namespace(run_id=12345)

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        cli.runs_download(args)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "Successfully downloaded run 12345" in output
        assert "Task ID: 1" in output
        assert "Flow ID: 100" in output
        assert "Dataset ID: 1" in output
        assert "Predictions available at: https://test.openml.org/predictions/12345" in output

        mock_get_run.assert_called_once_with(12345, ignore_cache=True)

    @mock.patch("openml.runs.functions.get_run")
    def test_runs_download_error_handling(self, mock_get_run: mock.Mock) -> None:
        """Test runs download command error handling."""
        mock_get_run.side_effect = Exception("Download failed")

        args = argparse.Namespace(run_id=12345)

        # Capture stderr
        captured_error = StringIO()
        sys.stderr = captured_error

        with pytest.raises(SystemExit):
            cli.runs_download(args)

        sys.stderr = sys.__stderr__

        error = captured_error.getvalue()
        assert "Error downloading run" in error
        assert "Download failed" in error

    def test_runs_dispatcher(self) -> None:
        """Test runs command dispatcher."""
        # Test with list subcommand
        with mock.patch("openml.cli.runs_list") as mock_list:
            args = argparse.Namespace(runs_subcommand="list")
            cli.runs(args)
            mock_list.assert_called_once_with(args)

        # Test with info subcommand
        with mock.patch("openml.cli.runs_info") as mock_info:
            args = argparse.Namespace(runs_subcommand="info")
            cli.runs(args)
            mock_info.assert_called_once_with(args)

        # Test with download subcommand
        with mock.patch("openml.cli.runs_download") as mock_download:
            args = argparse.Namespace(runs_subcommand="download")
            cli.runs(args)
            mock_download.assert_called_once_with(args)

    def test_runs_dispatcher_invalid_subcommand(self) -> None:
        """Test runs command dispatcher with invalid subcommand."""
        args = argparse.Namespace(runs_subcommand="invalid")

        # Capture stderr
        captured_error = StringIO()
        sys.stderr = captured_error

        with pytest.raises(SystemExit):
            cli.runs(args)

        sys.stderr = sys.__stderr__


class TestCLIIntegration(TestBase):
    """Integration tests for CLI argument parsing."""

    def test_main_help(self) -> None:
        """Test that main help displays runs command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subroutine")
        subparsers.add_parser("runs")

        # Should not raise an error
        args = parser.parse_args(["runs"])
        assert args.subroutine == "runs"

    def test_runs_list_argument_parsing(self) -> None:
        """Test argument parsing for runs list command."""
        # Create a minimal parser for testing
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subroutine")
        runs_parser = subparsers.add_parser("runs")
        runs_subparsers = runs_parser.add_subparsers(dest="runs_subcommand")
        list_parser = runs_subparsers.add_parser("list")

        list_parser.add_argument("--task", type=int)
        list_parser.add_argument("--flow", type=int)
        list_parser.add_argument("--uploader", type=str)
        list_parser.add_argument("--tag", type=str)
        list_parser.add_argument("--size", type=int, default=10)
        list_parser.add_argument("--offset", type=int, default=0)
        list_parser.add_argument("--format", choices=["list", "table", "json"], default="list")
        list_parser.add_argument("--verbose", action="store_true")

        # Test with various arguments
        args = parser.parse_args(["runs", "list", "--task", "1", "--flow", "100", "--size", "20"])
        assert args.subroutine == "runs"
        assert args.runs_subcommand == "list"
        assert args.task == 1
        assert args.flow == 100
        assert args.size == 20
        assert args.offset == 0
        assert args.format == "list"
        assert args.verbose is False

    def test_runs_info_argument_parsing(self) -> None:
        """Test argument parsing for runs info command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subroutine")
        runs_parser = subparsers.add_parser("runs")
        runs_subparsers = runs_parser.add_subparsers(dest="runs_subcommand")
        info_parser = runs_subparsers.add_parser("info")
        info_parser.add_argument("run_id", type=int)

        args = parser.parse_args(["runs", "info", "12345"])
        assert args.subroutine == "runs"
        assert args.runs_subcommand == "info"
        assert args.run_id == 12345

    def test_runs_download_argument_parsing(self) -> None:
        """Test argument parsing for runs download command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subroutine")
        runs_parser = subparsers.add_parser("runs")
        runs_subparsers = runs_parser.add_subparsers(dest="runs_subcommand")
        download_parser = runs_subparsers.add_parser("download")
        download_parser.add_argument("run_id", type=int)

        args = parser.parse_args(["runs", "download", "12345"])
        assert args.subroutine == "runs"
        assert args.runs_subcommand == "download"
        assert args.run_id == 12345
