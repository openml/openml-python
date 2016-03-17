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
    def test_get_cached_splits(self):
        raise Exception()

    @unittest.skip("Not implemented yet.")
    def test_get_cached_split(self):
        raise Exception()

    ############################################################################
    # Flows
    @unittest.skip('The method which is tested by this function doesnt exist')
    def test_download_flow_list(self):
        def check_flow(flow):
            self.assertIsInstance(flow, dict)
            self.assertEqual(len(flow), 6)

        flows = self.connector.get_flow_list()
        self.assertGreaterEqual(len(flows), 1448)
        for flow in flows:
            check_flow(flow)

    def test_upload_flow(self):
        description = ('''<oml:flow xmlns:oml="http://openml.org/openml"><oml:name>Test</oml:name>'''
                       '''<oml:description>description</oml:description> </oml:flow>''')
        return_code, return_value = self.connector.upload_flow(description, "Testing upload flow")
        # self.assertTrue("This is a read-only account" in return_value)
        self.assertEqual(return_code, 200)
