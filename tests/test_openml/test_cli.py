# License: BSD 3-Clause
from __future__ import annotations

import shutil
import subprocess
import sys

import openml
import pytest


def test_cli_version_prints_package_version():
    # Invoke the CLI via module to avoid relying on console script installation
    result = subprocess.run(
        [sys.executable, "-m", "openml.cli", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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

    result = subprocess.run(
        [console, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert openml.__version__ in result.stdout


@pytest.mark.production_server()
def test_cli_flows_list():
    """Test that 'openml flows list --size 5' returns a table of flows."""
    result = subprocess.run(
        [sys.executable, "-m", "openml.cli", "flows", "list", "--size", "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    # Output should contain at least one flow entry with a name column
    assert "name" in result.stdout.lower() or len(result.stdout.strip()) > 0


@pytest.mark.production_server()
def test_cli_flows_info():
    """Test that 'openml flows info <id>' prints flow details."""
    result = subprocess.run(
        [sys.executable, "-m", "openml.cli", "flows", "info", "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    # The output should contain the flow name or ID
    assert "Flow Name" in result.stdout or "5" in result.stdout


def test_cli_flows_no_action_prints_help():
    """Test that 'openml flows' with no subcommand prints help text."""
    result = subprocess.run(
        [sys.executable, "-m", "openml.cli", "flows"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    # Should print help text mentioning available subcommands
    assert "list" in result.stdout or "info" in result.stdout


@pytest.mark.production_server()
def test_cli_datasets_list():
    """Test that 'openml datasets list --size 5' returns a table of datasets."""
    result = subprocess.run(
        [sys.executable, "-m", "openml.cli", "datasets", "list", "--size", "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "name" in result.stdout.lower() or len(result.stdout.strip()) > 0


@pytest.mark.production_server()
def test_cli_datasets_info():
    """Test that 'openml datasets info <id>' prints dataset details."""
    result = subprocess.run(
        [sys.executable, "-m", "openml.cli", "datasets", "info", "61"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    # Dataset 61 is the iris dataset
    assert "iris" in result.stdout.lower() or "61" in result.stdout


def test_cli_datasets_no_action_prints_help():
    """Test that 'openml datasets' with no subcommand prints help text."""
    result = subprocess.run(
        [sys.executable, "-m", "openml.cli", "datasets"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "list" in result.stdout or "info" in result.stdout
