# License: BSD 3-Clause
from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile
import unittest.mock
from copy import copy
from typing import Any, Iterator
from pathlib import Path
import platform

import pytest

import openml.config
import openml.testing
from openml.testing import TestBase


@contextmanager
def safe_environ_patcher(key: str, value: Any) -> Iterator[None]:
    """Context manager to temporarily set an environment variable.

    Safe to errors happening in the yielded to function.
    """
    _prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    except Exception as e:
        raise e
    finally:
        os.environ.pop(key)
        if _prev is not None:
            os.environ[key] = _prev


class TestConfig(openml.testing.TestBase):
    @unittest.mock.patch("openml.config.openml_logger.warning")
    @unittest.mock.patch("openml.config._create_log_handlers")
    @unittest.skipIf(os.name == "nt", "https://github.com/openml/openml-python/issues/1033")
    @unittest.skipIf(
        platform.uname().release.endswith(("-Microsoft", "microsoft-standard-WSL2")),
        "WSL does nto support chmod as we would need here, see https://github.com/microsoft/WSL/issues/81",
    )
    def test_non_writable_home(self, log_handler_mock, warnings_mock):
        with tempfile.TemporaryDirectory(dir=self.workdir) as td:
            os.chmod(td, 0o444)
            _dd = copy(openml.config._defaults)
            _dd["cachedir"] = Path(td) / "something-else"
            openml.config._setup(_dd)

        assert warnings_mock.call_count == 1
        assert log_handler_mock.call_count == 1
        assert not log_handler_mock.call_args_list[0][1]["create_file_handler"]
        assert openml.config._root_cache_directory == Path(td) / "something-else"

    @unittest.skipIf(platform.system() != "Linux", "XDG only exists for Linux systems.")
    def test_XDG_directories_do_not_exist(self):
        with tempfile.TemporaryDirectory(dir=self.workdir) as td:
            # Save previous state
            path = Path(td) / "fake_xdg_cache_home"
            with safe_environ_patcher("XDG_CONFIG_HOME", str(path)):
                expected_config_dir = path / "openml"
                expected_determined_config_file_path = expected_config_dir / "config"

                # Ensure that it correctly determines the path to the config file
                determined_config_file_path = openml.config.determine_config_file_path()
                assert determined_config_file_path == expected_determined_config_file_path

                # Ensure that setup will create the config folder as the configuration
                # will be written to that location.
                openml.config._setup()
                assert expected_config_dir.exists()

    def test_get_config_as_dict(self):
        """Checks if the current configuration is returned accurately as a dict."""
        config = openml.config.get_config_as_dict()
        _config = {}
        _config["apikey"] = TestBase.user_key
        _config["server"] = "https://test.openml.org/api/v1/xml"
        _config["cachedir"] = self.workdir
        _config["avoid_duplicate_runs"] = False
        _config["connection_n_retries"] = 20
        _config["retry_policy"] = "robot"
        _config["show_progress"] = False
        assert isinstance(config, dict)
        assert len(config) == 7
        self.assertDictEqual(config, _config)

    def test_setup_with_config(self):
        """Checks if the OpenML configuration can be updated using _setup()."""
        _config = {}
        _config["apikey"] = TestBase.user_key
        _config["server"] = "https://www.openml.org/api/v1/xml"
        _config["cachedir"] = self.workdir
        _config["avoid_duplicate_runs"] = True
        _config["retry_policy"] = "human"
        _config["connection_n_retries"] = 100
        _config["show_progress"] = False
        orig_config = openml.config.get_config_as_dict()
        openml.config._setup(_config)
        updated_config = openml.config.get_config_as_dict()
        openml.config._setup(orig_config)  # important to not affect other unit tests
        self.assertDictEqual(_config, updated_config)


class TestConfigurationForExamples(openml.testing.TestBase):
    @pytest.mark.production()
    def test_switch_to_example_configuration(self):
        """Verifies the test configuration is loaded properly."""
        # Below is the default test key which would be used anyway, but just for clarity:
        openml.config.apikey = "any-api-key"
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()

        assert openml.config.apikey == TestBase.user_key
        assert openml.config.server == self.test_server

    @pytest.mark.production()
    def test_switch_from_example_configuration(self):
        """Verifies the previous configuration is loaded after stopping."""
        # Below is the default test key which would be used anyway, but just for clarity:
        openml.config.apikey = TestBase.user_key
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()
        openml.config.stop_using_configuration_for_example()

        assert openml.config.apikey == TestBase.user_key
        assert openml.config.server == self.production_server

    def test_example_configuration_stop_before_start(self):
        """Verifies an error is raised if `stop_...` is called before `start_...`."""
        error_regex = ".*stop_use_example_configuration.*start_use_example_configuration.*first"
        # Tests do not reset the state of this class. Thus, we ensure it is in
        # the original state before the test.
        openml.config.ConfigurationForExamples._start_last_called = False
        self.assertRaisesRegex(
            RuntimeError,
            error_regex,
            openml.config.stop_using_configuration_for_example,
        )

    @pytest.mark.production()
    def test_example_configuration_start_twice(self):
        """Checks that the original config can be returned to if `start..` is called twice."""
        openml.config.apikey = TestBase.user_key
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()
        openml.config.start_using_configuration_for_example()
        openml.config.stop_using_configuration_for_example()

        assert openml.config.apikey == TestBase.user_key
        assert openml.config.server == self.production_server


def test_configuration_file_not_overwritten_on_load():
    """Regression test for #1337"""
    config_file_content = "apikey = abcd"
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file_path = Path(tmpdir) / "config"
        with config_file_path.open("w") as config_file:
            config_file.write(config_file_content)

        read_config = openml.config._parse_config(config_file_path)

        with config_file_path.open("r") as config_file:
            new_file_content = config_file.read()

    assert config_file_content == new_file_content
    assert "abcd" == read_config["apikey"]


def test_configuration_loads_booleans(tmp_path):
    config_file_content = "avoid_duplicate_runs=true\nshow_progress=false"
    tmp_file = tmp_path / "config"
    with tmp_file.open("w") as config_file:
        config_file.write(config_file_content)
    read_config = openml.config._parse_config(tmp_file)

    # Explicit test to avoid truthy/falsy modes of other types
    assert read_config["avoid_duplicate_runs"] is True
    assert read_config["show_progress"] is False


def test_openml_cache_dir_env_var(tmp_path: Path) -> None:
    expected_path = tmp_path / "test-cache"

    with safe_environ_patcher("OPENML_CACHE_DIR", str(expected_path)):
        openml.config._setup()
        assert openml.config._root_cache_directory == expected_path
        assert openml.config.get_cache_directory() == str(expected_path / "org" / "openml" / "www")
