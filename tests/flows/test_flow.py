import hashlib
import re
import sys
import time
import unittest

import xmltodict

import sklearn.dummy
import sklearn.ensemble
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree

from openml.testing import TestBase
from openml._api_calls import _perform_api_call
import openml
from openml.flows.sklearn import SklearnToFlowConverter

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

    def test_get_flow(self):
        # We need to use the production server here because 4024 is not the test
        # server
        openml.config.server = self.production_server

        flow = openml.flows.get_flow(4024)
        self.assertIsInstance(flow, openml.OpenMLFlow)
        self.assertEqual(flow.flow_id, 4024)
        self.assertEqual(len(flow.parameters), 24)
        self.assertEqual(len(flow.components), 1)

        subflow_1 = list(flow.components.values())[0]
        self.assertIsInstance(subflow_1, openml.OpenMLFlow)
        self.assertEqual(subflow_1.flow_id, 4025)
        self.assertEqual(len(subflow_1.parameters), 14)
        self.assertEqual(subflow_1.parameters['E'], 'CC')
        self.assertEqual(len(subflow_1.components), 1)

        subflow_2 = list(subflow_1.components.values())[0]
        self.assertIsInstance(subflow_2, openml.OpenMLFlow)
        self.assertEqual(subflow_2.flow_id, 4026)
        self.assertEqual(len(subflow_2.parameters), 13)
        self.assertEqual(subflow_2.parameters['I'], '10')
        self.assertEqual(len(subflow_2.components), 1)

        subflow_3 = list(subflow_2.components.values())[0]
        self.assertIsInstance(subflow_3, openml.OpenMLFlow)
        self.assertEqual(subflow_3.flow_id, 1724)
        self.assertEqual(len(subflow_3.parameters), 11)
        self.assertEqual(subflow_3.parameters['L'], '-1')
        self.assertEqual(len(subflow_3.components), 0)

    def test_from_xml_to_xml(self):
        # Get the raw xml thing
        # TODO maybe get this via get_flow(), which would have to be refactored to allow getting only the xml dictionary
        for flow_id in [1185, 1244, 1196, 1112, ]:
            flow_xml = _perform_api_call("flow/%d" % flow_id)[1]
            flow_dict = xmltodict.parse(flow_xml)

            flow = openml.OpenMLFlow._from_xml(flow_dict)
            new_xml = flow._to_xml()

            flow_xml = flow_xml.replace('  ', '').replace('\t', '').strip().replace('\n\n', '\n').replace('&quot;', '"')
            flow_xml = re.sub(r'^$', '', flow_xml)
            new_xml = new_xml.replace('  ', '').replace('\t', '').strip().replace('\n\n', '\n').replace('&quot;', '"')
            new_xml = re.sub(r'^$', '', new_xml)
            with open('../new.xml', 'w') as fh:
                fh.write(new_xml)
            with open('../old.xml', 'w') as fh:
                fh.write(flow_xml)

            self.assertEqual(new_xml, flow_xml)

    def test_to_xml_from_xml(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier())
        model = sklearn.pipeline.Pipeline(steps=(
            ('scaler', scaler), ('boosting', boosting)))
        flow = openml.flows.create_flow_from_model(model, SklearnToFlowConverter())
        flow.flow_id = -234
        # end of setup

        xml = flow._to_xml()
        xml_dict = xmltodict.parse(xml)
        new_flow = openml.flows.OpenMLFlow._from_xml(xml_dict)
        self.assertEqual(new_flow, flow)
        self.assertIsNot(new_flow, flow)

    @mock.patch.object(openml.OpenMLFlow, '_get_name', autospec=True)
    def test_upload_flow(self, name_mock):
        flow = openml.OpenMLFlow(name='Test',
                                 description="test description",
                                 model=sklearn.dummy.DummyClassifier())

        # Create a unique prefix for the flow. Necessary because the flow is
        # identified by its name and external version online. Having a unique
        #  name allows us to publish the same flow in each test run
        md5 = hashlib.md5()
        md5.update(str(time.time()).encode('utf-8'))
        sentinel = md5.hexdigest()[:10]
        name_mock.return_value = '%s%s' % (sentinel, flow.name)

        flow.publish()
        self.assertIsInstance(flow.flow_id, int)
