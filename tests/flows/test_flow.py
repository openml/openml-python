import unittest
from sklearn.dummy import DummyClassifier

from openml.testing import TestBase
import openml


class TestFlow(TestBase):
    @unittest.skip('The method which is tested by this function doesnt exist')
    def test_download_flow_list(self):
        def check_flow(flow):
            self.assertIsInstance(flow, dict)
            self.assertEqual(len(flow), 6)

        flows = openml.flows.get_flow_list()
        self.assertGreaterEqual(len(flows), 1448)
        for flow in flows:
            check_flow(flow)

    @unittest.skip('Not tested until test sentinels are added back.')
    def test_upload_flow(self):
        flow = openml.OpenMLFlow(model=DummyClassifier(), description="test description")
        return_code, return_value = flow.publish()
        # self.assertTrue("This is a read-only account" in return_value)
        self.assertEqual(return_code, 200)
