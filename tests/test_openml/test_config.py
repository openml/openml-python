import os

import openml.config
import openml.testing


class TestConfig(openml.testing.TestBase):

    def test_config_loading(self):
        self.assertTrue(os.path.exists(openml.config.config_file))
        self.assertTrue(os.path.isdir(os.path.expanduser('~/.openml')))
