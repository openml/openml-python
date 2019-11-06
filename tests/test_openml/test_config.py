# License: BSD 3-Clause

import os

import openml.config
import openml.testing


class TestConfig(openml.testing.TestBase):

    def test_config_loading(self):
        self.assertTrue(os.path.exists(openml.config.config_file))
        self.assertTrue(os.path.isdir(os.path.expanduser('~/.openml')))


class TestConfigurationForExamples(openml.testing.TestBase):

    def test_switch_to_example_configuration(self):
        """ Verifies the test configuration is loaded properly. """
        # Below is the default test key which would be used anyway, but just for clarity:
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()

        self.assertEqual(openml.config.apikey, "c0c42819af31e706efe1f4b88c23c6c1")
        self.assertEqual(openml.config.server, self.test_server)

    def test_switch_from_example_configuration(self):
        """ Verifies the previous configuration is loaded after stopping. """
        # Below is the default test key which would be used anyway, but just for clarity:
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()
        openml.config.stop_using_configuration_for_example()

        self.assertEqual(openml.config.apikey, "610344db6388d9ba34f6db45a3cf71de")
        self.assertEqual(openml.config.server, self.production_server)

    def test_example_configuration_stop_before_start(self):
        """ Verifies an error is raised is `stop_...` is called before `start_...`. """
        error_regex = ".*stop_use_example_configuration.*start_use_example_configuration.*first"
        self.assertRaisesRegex(RuntimeError, error_regex,
                               openml.config.stop_using_configuration_for_example)

    def test_example_configuration_start_twice(self):
        """ Checks that the original config can be returned to if `start..` is called twice. """
        openml.config.apikey = "610344db6388d9ba34f6db45a3cf71de"
        openml.config.server = self.production_server

        openml.config.start_using_configuration_for_example()
        openml.config.start_using_configuration_for_example()
        openml.config.stop_using_configuration_for_example()

        self.assertEqual(openml.config.apikey, "610344db6388d9ba34f6db45a3cf71de")
        self.assertEqual(openml.config.server, self.production_server)
