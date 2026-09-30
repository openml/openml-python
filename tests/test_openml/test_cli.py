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
