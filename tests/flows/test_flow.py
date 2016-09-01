import sys
import unittest

from sklearn.dummy import DummyClassifier

from openml.testing import TestBase
import openml

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock


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

    @mock.patch.object(openml.OpenMLFlow, '_get_name', autospec=True)
    def test_upload_flow(self, name_mock):
        flow = openml.OpenMLFlow(model=DummyClassifier(), description="test description")
        name_mock.return_value = '%s%s' % (self.sentinel, flow.name)
        flow.publish()
        self.assertIsInstance(flow.flow_id, int)
