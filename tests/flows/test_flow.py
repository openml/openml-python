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

    def test_upload_flow(self):
        flow = openml.OpenMLFlow(model=DummyClassifier(), description="test description")
        return_code, return_value = flow.publish()

        self.assertEqual(return_code, 200)
    def test_get_flow(self):
        flow = openml.flows.get_flow(1185)
        self.assertIsInstance(flow, openml.OpenMLFlow)
        self.assertEqual(flow.id, 1185)
        self.assertEqual(len(flow.parameters), 14)
        for parameter in flow.parameters:
            self.assertEqual(len(parameter), 4)
        self.assertEqual(len(flow.components), 1)
        for component in flow.components:
            self.assertEqual(len(component), 2)
            self.assertIsInstance(component['flow'], openml.OpenMLFlow)
