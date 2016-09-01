import hashlib
import sys
import time
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

        # Create a unique prefix for the flow. Necessary because the flow is
        # identified by its name and external version online. Having a unique
        #  name allows us to publish the same flow in each test run
        md5 = hashlib.md5()
        md5.update(str(time.time()).encode('utf-8'))
        sentinel = md5.hexdigest()[:10]
        name_mock.return_value = '%s%s' % (sentinel, flow.name)

        flow.publish()
        self.assertIsInstance(flow.flow_id, int)
