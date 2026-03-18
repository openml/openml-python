# License: BSD 3-Clause
from __future__ import annotations

import shutil
import subprocess
import sys
from unittest import mock

import pytest

import openml
from openml.cli import main


def test_cli_version_prints_package_version():
    # Invoke the CLI via module to avoid relying on console script installation
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "openml.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Ensure successful exit and version present in stdout only
    assert result.returncode == 0
    assert result.stderr == ""
    assert openml.__version__ in result.stdout


def test_console_script_version_prints_package_version():
    # Try to locate the console script; skip if not installed in PATH
    console = shutil.which("openml")
    if console is None:
        pytest.skip("'openml' console script not found in PATH")

    result = subprocess.run(  # noqa: S603
        [console, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert openml.__version__ in result.stdout


def test_upload_dataset_arg_parsing():
    # Test that the dataset subcommand correctly parses required and optional arguments
    test_args = [
        "upload", "dataset", "data.csv",
        "--name", "MyDataset",
        "--description", "A test dataset",
        "--default_target_attribute", "target",
        "--creator", "TestUser",
    ]
    with (
        mock.patch("sys.argv", ["openml", *test_args]),
        mock.patch("openml.cli.upload") as mock_upload,
    ):
        main()
        args = mock_upload.call_args[0][0]
        assert args.subroutine == "upload"
        assert args.upload_resource == "dataset"
        assert args.file_path == "data.csv"
        assert args.name == "MyDataset"
        assert args.description == "A test dataset"
        assert args.default_target_attribute == "target"
        assert args.creator == "TestUser"
        assert args.contributor is None
        assert args.licence is None


def test_upload_flow_arg_parsing():
    # Test that the flow subcommand correctly parses positional and optional arguments
    test_args = ["upload", "flow", "model.pkl", "--name", "MyFlow", "--description", "A flow"]
    with (
        mock.patch("sys.argv", ["openml", *test_args]),
        mock.patch("openml.cli.upload") as mock_upload,
    ):
        main()
        args = mock_upload.call_args[0][0]
        assert args.upload_resource == "flow"
        assert args.file_path == "model.pkl"
        assert args.name == "MyFlow"
        assert args.description == "A flow"


def test_upload_run_arg_parsing():
    # Test that the run subcommand correctly parses positional and flag arguments
    test_args = ["upload", "run", "/path/to/run_dir", "--no_model"]
    with (
        mock.patch("sys.argv", ["openml", *test_args]),
        mock.patch("openml.cli.upload") as mock_upload,
    ):
        main()
        args = mock_upload.call_args[0][0]
        assert args.upload_resource == "run"
        assert args.file_path == "/path/to/run_dir"
        assert args.no_model is True


def test_upload_run_no_model_defaults_false():
    # Test that the --no_model flag defaults to False if not provided
    test_args = ["upload", "run", "/path/to/run_dir"]
    with (
        mock.patch("sys.argv", ["openml", *test_args]),
        mock.patch("openml.cli.upload") as mock_upload,
    ):
        main()
        args = mock_upload.call_args[0][0]
        assert args.no_model is False
