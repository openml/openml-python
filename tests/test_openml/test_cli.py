# License: BSD 3-Clause
from __future__ import annotations

import shutil
import subprocess
import sys
from unittest import mock

import pytest

import openml
from openml.cli import main, upload, upload_dataset, upload_flow, upload_run


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


def test_upload_dataset_csv(tmp_path):
    # Verify CSV upload calls create_dataset and publish with correct arguments
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col_a,col_b,target\n1,2.0,cat\n3,4.0,dog\n")

    args = mock.MagicMock()
    args.file_path = str(csv_file)
    args.name = "TestDS"
    args.description = "desc"
    args.default_target_attribute = "target"
    for attr in (
        "creator", "contributor", "collection_date", "language", "licence",
        "ignore_attribute", "citation", "row_id_attribute",
        "original_data_url", "paper_url", "version_label", "update_comment",
    ):
        setattr(args, attr, None)

    mock_ds = mock.MagicMock(id=42, openml_url="https://openml.org/d/42")
    with mock.patch("openml.datasets.create_dataset", return_value=mock_ds) as mock_create:
        upload_dataset(args)
        mock_create.assert_called_once()
        assert mock_create.call_args[1]["name"] == "TestDS"
        assert mock_create.call_args[1]["attributes"] == "auto"
        mock_ds.publish.assert_called_once()


def test_upload_dataset_file_not_found(capsys):
    # Verify a clear error is shown when the file does not exist
    args = mock.MagicMock(file_path="/nonexistent/data.csv")
    with pytest.raises(SystemExit, match="1"):
        upload_dataset(args)
    assert "not found" in capsys.readouterr().out


def test_upload_dataset_unsupported_format(tmp_path, capsys):
    # Verify unsupported file extensions are rejected
    bad_file = tmp_path / "data.json"
    bad_file.write_text("{}")
    args = mock.MagicMock(file_path=str(bad_file))
    with pytest.raises(SystemExit, match="1"):
        upload_dataset(args)
    assert "Unsupported file format" in capsys.readouterr().out


def test_upload_flow_uses_extension_api(tmp_path):
    # Verify upload_flow uses get_extension_by_model instead of direct openml_sklearn import
    pkl_file = tmp_path / "model.pkl"
    pkl_file.write_bytes(b"fake")

    mock_model = mock.MagicMock()
    mock_flow = mock.MagicMock(flow_id=99, openml_url="https://openml.org/f/99")
    mock_ext = mock.MagicMock()
    mock_ext.model_to_flow.return_value = mock_flow

    args = mock.MagicMock(file_path=str(pkl_file), name=None, description=None)
    with (
        mock.patch("pickle.load", return_value=mock_model),
        mock.patch(
            "openml.extensions.get_extension_by_model",
            return_value=mock_ext,
        ) as mock_get_ext,
    ):
        upload_flow(args)
        mock_get_ext.assert_called_once_with(mock_model, raise_if_no_extension=True)
        mock_flow.publish.assert_called_once()


def test_upload_flow_file_not_found(capsys):
    # Verify a clear error is shown when the pickle file does not exist
    args = mock.MagicMock(file_path="/nonexistent/model.pkl")
    with pytest.raises(SystemExit, match="1"):
        upload_flow(args)
    assert "not found" in capsys.readouterr().out


def test_upload_run_calls_from_filesystem(tmp_path):
    # Verify upload_run delegates to OpenMLRun.from_filesystem and publishes
    run_dir = tmp_path / "run_output"
    run_dir.mkdir()

    mock_run = mock.MagicMock(run_id=55, openml_url="https://openml.org/r/55")
    args = mock.MagicMock(file_path=str(run_dir), no_model=False)

    with mock.patch.object(
        openml.runs.OpenMLRun, "from_filesystem", return_value=mock_run,
    ) as mock_fs:
        upload_run(args)
        mock_fs.assert_called_once_with(run_dir, expect_model=True)
        mock_run.publish.assert_called_once()


def test_upload_run_dir_not_found(capsys):
    # Verify a clear error is shown when the run directory does not exist
    args = mock.MagicMock(file_path="/nonexistent/run_dir", no_model=False)
    with pytest.raises(SystemExit, match="1"):
        upload_run(args)
    assert "not found" in capsys.readouterr().out


def test_upload_missing_api_key(capsys):
    # Verify upload refuses to proceed without an API key
    args = mock.MagicMock(upload_resource="dataset")
    with (
        openml.config.overwrite_config_context({"apikey": ""}),
        pytest.raises(SystemExit, match="1"),
    ):
        upload(args)
    assert "No API key configured" in capsys.readouterr().out
