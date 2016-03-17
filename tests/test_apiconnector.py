import unittest
from openml.testing import TestBase


class TestAPIConnector(TestBase):
    """Test the APIConnector

    Note
    ----
    A config file with the username and password must be present to test the
    API calls.
    """

    ############################################################################
    # Test administrative stuff
    @unittest.skip("Not implemented yet.")
    def test_parse_config(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test_get_cached_tasks(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test_get_cached_task(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test__get_cached_splits(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test__get_cached_split(self):
        raise Exception()

