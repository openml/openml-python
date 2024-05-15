# License: BSD 3-Clause
from __future__ import annotations

import os
import tempfile
import unittest.mock
from copy import copy
from pathlib import Path

import pytest

import openml.config
import openml.testing


class TestConfig(openml.testing.TestBase):
    @unittest.mock.patch("openml.config.openml_logger.warning")
    @unittest.mock.patch("openml.config._create_log_handlers")
    @unittest.skipIf(os.name == "nt", "https://github.com/openml/openml-python/issues/1033")
    def test_non_writable_home(self, log_handler_mock, warnings_mock):
        with tempfile.TemporaryDirectory(dir=self.workdir) as td:
            os.chmod(td, 0o444)
            _dd = copy(openml.config._defaults)
            _dd["cachedir"] = Path(td) / "something-else"
            openml.config._setup(_dd)

        assert warnings_mock.call_count == 2
        assert log_handler_mock.call_count == 1
        assert not log_handler_mock.call_args_list[0][1]["create_file_handler"]
        assert openml.config._root_cache_directory == Path(td) / "something-else"

    @unittest.mock.patch("os.path.expanduser")
    def test_XDG_directories_do_not_exist(self, expanduser_mock):
        with tempfile.TemporaryDirectory(dir=self.workdir) as td:

            def side_effect(path_):
                return os.path.join(td, str(path_).replace("~/", ""))

            expanduser_mock.side_effect = side_effect
            openml.config._setup()

    def test_get_config_as_dict(self):
        """Checks if the current configuration is returned accurately as a dict."""
        config = openml.config.get_config_as_dict()
        _config = {}
        _config["apikey"] = "610344db6388d9ba34f6db45a3cf71de"
        _config["server"] = "https://test.openml.org/api/v1/xml"
        _config["cachedir"] = self.workdir
        _config["avoid_duplicate_runs"] = False
        _config["connection_n_retries"] = 20
        _config["retry_policy"] = "robot"
        assert isinstance(config, dict)
        assert len(config) == 6
        self.assertDictEqual(config, _config)

    def test_setup_with_config(self):
        """Checks if the OpenML configuration can be updated using _setup()."""
        _config = {}
        _config["apikey"] = "610344db6388d9ba34f6db45a3cf71de"
        _config["server"] = "https://www.openml.org/api/v1/xml"
        _config["cachedir"] = self.workdir
        _config["avoid_duplicate_runs"] = True
        _config["retry_policy"] = "human"
        _config["connection_n_retries"] = 100
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
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()

        assert openml.config.apikey == "c0c42819af31e706efe1f4b88c23c6c1"
        assert openml.config.server == self.test_server

    @pytest.mark.production()
    def test_switch_from_example_configuration(self):
        """Verifies the previous configuration is loaded after stopping."""
        # Below is the default test key which would be used anyway, but just for clarity:
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()
        openml.config.stop_using_configuration_for_example()

        assert openml.config.apikey == "610344db6388d9ba34f6db45a3cf71de"
        assert openml.config.server == self.production_server

    def test_example_configuration_stop_before_start(self):
        """Verifies an error is raised is `stop_...` is called before `start_...`."""
        error_regex = ".*stop_use_example_configuration.*start_use_example_configuration.*first"
        self.assertRaisesRegex(
            RuntimeError,
            error_regex,
            openml.config.stop_using_configuration_for_example,
        )

    @pytest.mark.production()
    def test_example_configuration_start_twice(self):
        """Checks that the original config can be returned to if `start..` is called twice."""
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()
        openml.config.start_using_configuration_for_example()
        openml.config.stop_using_configuration_for_example()

        assert openml.config.apikey == "610344db6388d9ba34f6db45a3cf71de"
        assert openml.config.server == self.production_server


def test_configuration_file_not_overwritten_on_load():
    """ Regression test for #1337 """
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
