# License: BSD 3-Clause

from collections import OrderedDict
import copy
import unittest

from distutils.version import LooseVersion
import sklearn
from sklearn import ensemble
import pandas as pd

import openml
from openml.testing import TestBase
import openml.extensions.sklearn


class TestFlowFunctions(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super(TestFlowFunctions, self).setUp()

    def tearDown(self):
        super(TestFlowFunctions, self).tearDown()

    def _check_flow(self, flow):
        self.assertEqual(type(flow), dict)
        self.assertEqual(len(flow), 6)
        self.assertIsInstance(flow['id'], int)
        self.assertIsInstance(flow['name'], str)
        self.assertIsInstance(flow['full_name'], str)
        self.assertIsInstance(flow['version'], str)
        # There are some runs on openml.org that can have an empty external version
        ext_version_str_or_none = (isinstance(flow['external_version'], str)
                                   or flow['external_version'] is None)
        self.assertTrue(ext_version_str_or_none)

    def test_list_flows(self):
        openml.config.server = self.production_server
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        flows = openml.flows.list_flows()
        # 3000 as the number of flows on openml.org
        self.assertGreaterEqual(len(flows), 1500)
        for fid in flows:
            self._check_flow(flows[fid])

    def test_list_flows_output_format(self):
        openml.config.server = self.production_server
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        flows = openml.flows.list_flows(output_format='dataframe')
        self.assertIsInstance(flows, pd.DataFrame)
        self.assertGreaterEqual(len(flows), 1500)

    def test_list_flows_empty(self):
        openml.config.server = self.production_server
        flows = openml.flows.list_flows(tag='NoOneEverUsesThisTag123')
        if len(flows) > 0:
            raise ValueError(
                'UnitTest Outdated, got somehow results (please adapt)'
            )

        self.assertIsInstance(flows, dict)

    def test_list_flows_by_tag(self):
        openml.config.server = self.production_server
        flows = openml.flows.list_flows(tag='weka')
        self.assertGreaterEqual(len(flows), 5)
        for did in flows:
            self._check_flow(flows[did])

    def test_list_flows_paginate(self):
        openml.config.server = self.production_server
        size = 10
        maximum = 100
        for i in range(0, maximum, size):
            flows = openml.flows.list_flows(offset=i, size=size)
            self.assertGreaterEqual(size, len(flows))
            for did in flows:
                self._check_flow(flows[did])

    def test_are_flows_equal(self):
        flow = openml.flows.OpenMLFlow(name='Test',
                                       description='Test flow',
                                       model=None,
                                       components=OrderedDict(),
                                       parameters=OrderedDict(),
                                       parameters_meta_info=OrderedDict(),
                                       external_version='1',
                                       tags=['abc', 'def'],
                                       language='English',
                                       dependencies='abc',
                                       class_name='Test',
                                       custom_name='Test')

        # Test most important values that can be set by a user
        openml.flows.functions.assert_flows_equal(flow, flow)
        for attribute, new_value in [('name', 'Tes'),
                                     ('external_version', '2'),
                                     ('language', 'english'),
                                     ('dependencies', 'ab'),
                                     ('class_name', 'Tes'),
                                     ('custom_name', 'Tes')]:
            new_flow = copy.deepcopy(flow)
            setattr(new_flow, attribute, new_value)
            self.assertNotEqual(
                getattr(flow, attribute),
                getattr(new_flow, attribute),
            )
            self.assertRaises(
                ValueError,
                openml.flows.functions.assert_flows_equal,
                flow,
                new_flow,
            )

        # Test that the API ignores several keys when comparing flows
        openml.flows.functions.assert_flows_equal(flow, flow)
        for attribute, new_value in [('flow_id', 1),
                                     ('uploader', 1),
                                     ('version', 1),
                                     ('upload_date', '18.12.1988'),
                                     ('binary_url', 'openml.org'),
                                     ('binary_format', 'gzip'),
                                     ('binary_md5', '12345'),
                                     ('model', []),
                                     ('tags', ['abc', 'de'])]:
            new_flow = copy.deepcopy(flow)
            setattr(new_flow, attribute, new_value)
            self.assertNotEqual(
                getattr(flow, attribute),
                getattr(new_flow, attribute),
            )
            openml.flows.functions.assert_flows_equal(flow, new_flow)

        # Now test for parameters
        flow.parameters['abc'] = 1.0
        flow.parameters['def'] = 2.0
        openml.flows.functions.assert_flows_equal(flow, flow)
        new_flow = copy.deepcopy(flow)
        new_flow.parameters['abc'] = 3.0
        self.assertRaises(ValueError, openml.flows.functions.assert_flows_equal,
                          flow, new_flow)

        # Now test for components (subflows)
        parent_flow = copy.deepcopy(flow)
        subflow = copy.deepcopy(flow)
        parent_flow.components['subflow'] = subflow
        openml.flows.functions.assert_flows_equal(parent_flow, parent_flow)
        self.assertRaises(ValueError,
                          openml.flows.functions.assert_flows_equal,
                          parent_flow, subflow)
        new_flow = copy.deepcopy(parent_flow)
        new_flow.components['subflow'].name = 'Subflow name'
        self.assertRaises(ValueError,
                          openml.flows.functions.assert_flows_equal,
                          parent_flow, new_flow)

    def test_are_flows_equal_ignore_parameter_values(self):
        paramaters = OrderedDict((('a', 5), ('b', 6)))
        parameters_meta_info = OrderedDict((('a', None), ('b', None)))

        flow = openml.flows.OpenMLFlow(
            name='Test',
            description='Test flow',
            model=None,
            components=OrderedDict(),
            parameters=paramaters,
            parameters_meta_info=parameters_meta_info,
            external_version='1',
            tags=['abc', 'def'],
            language='English',
            dependencies='abc',
            class_name='Test',
            custom_name='Test',
        )

        openml.flows.functions.assert_flows_equal(flow, flow)
        openml.flows.functions.assert_flows_equal(flow, flow,
                                                  ignore_parameter_values=True)

        new_flow = copy.deepcopy(flow)
        new_flow.parameters['a'] = 7
        self.assertRaisesRegex(
            ValueError,
            r"values for attribute 'parameters' differ: "
            r"'OrderedDict\(\[\('a', 5\), \('b', 6\)\]\)'\nvs\n"
            r"'OrderedDict\(\[\('a', 7\), \('b', 6\)\]\)'",
            openml.flows.functions.assert_flows_equal,
            flow, new_flow,
        )
        openml.flows.functions.assert_flows_equal(flow, new_flow,
                                                  ignore_parameter_values=True)

        del new_flow.parameters['a']
        self.assertRaisesRegex(
            ValueError,
            r"values for attribute 'parameters' differ: "
            r"'OrderedDict\(\[\('a', 5\), \('b', 6\)\]\)'\nvs\n"
            r"'OrderedDict\(\[\('b', 6\)\]\)'",
            openml.flows.functions.assert_flows_equal,
            flow, new_flow,
        )
        self.assertRaisesRegex(
            ValueError,
            r"Flow Test: parameter set of flow differs from the parameters "
            r"stored on the server.",
            openml.flows.functions.assert_flows_equal,
            flow, new_flow, ignore_parameter_values=True,
        )

    def test_are_flows_equal_ignore_if_older(self):
        paramaters = OrderedDict((('a', 5), ('b', 6)))
        parameters_meta_info = OrderedDict((('a', None), ('b', None)))
        flow_upload_date = '2017-01-31T12-01-01'
        assert_flows_equal = openml.flows.functions.assert_flows_equal

        flow = openml.flows.OpenMLFlow(name='Test',
                                       description='Test flow',
                                       model=None,
                                       components=OrderedDict(),
                                       parameters=paramaters,
                                       parameters_meta_info=parameters_meta_info,
                                       external_version='1',
                                       tags=['abc', 'def'],
                                       language='English',
                                       dependencies='abc',
                                       class_name='Test',
                                       custom_name='Test',
                                       upload_date=flow_upload_date)

        assert_flows_equal(flow, flow, ignore_parameter_values_on_older_children=flow_upload_date)
        assert_flows_equal(flow, flow, ignore_parameter_values_on_older_children=None)
        new_flow = copy.deepcopy(flow)
        new_flow.parameters['a'] = 7
        self.assertRaises(ValueError, assert_flows_equal, flow, new_flow,
                          ignore_parameter_values_on_older_children=flow_upload_date)
        self.assertRaises(ValueError, assert_flows_equal, flow, new_flow,
                          ignore_parameter_values_on_older_children=None)

        new_flow.upload_date = '2016-01-31T12-01-01'
        self.assertRaises(ValueError, assert_flows_equal, flow, new_flow,
                          ignore_parameter_values_on_older_children=flow_upload_date)
        assert_flows_equal(flow, flow, ignore_parameter_values_on_older_children=None)

    @unittest.skipIf(LooseVersion(sklearn.__version__) < "0.20",
                     reason="OrdinalEncoder introduced in 0.20. "
                            "No known models with list of lists parameters in older versions.")
    def test_sklearn_to_flow_list_of_lists(self):
        from sklearn.preprocessing import OrdinalEncoder
        ordinal_encoder = OrdinalEncoder(categories=[[0, 1], [0, 1]])
        extension = openml.extensions.sklearn.SklearnExtension()

        # Test serialization works
        flow = extension.model_to_flow(ordinal_encoder)

        # Test flow is accepted by server
        self._add_sentinel_to_flow_name(flow)
        flow.publish()
        TestBase._mark_entity_for_removal('flow', (flow.flow_id, flow.name))
        TestBase.logger.info("collected from {}: {}".format(__file__.split('/')[-1], flow.flow_id))
        # Test deserialization works
        server_flow = openml.flows.get_flow(flow.flow_id, reinstantiate=True)
        self.assertEqual(server_flow.parameters['categories'], '[[0, 1], [0, 1]]')
        self.assertEqual(server_flow.model.categories, flow.model.categories)

    def test_get_flow1(self):
        # Regression test for issue #305
        # Basically, this checks that a flow without an external version can be loaded
        openml.config.server = self.production_server
        flow = openml.flows.get_flow(1)
        self.assertIsNone(flow.external_version)

    def test_get_flow_reinstantiate_model(self):
        model = ensemble.RandomForestClassifier(n_estimators=33)
        extension = openml.extensions.get_extension_by_model(model)
        flow = extension.model_to_flow(model)
        flow.publish(raise_error_if_exists=False)
        TestBase._mark_entity_for_removal('flow', (flow.flow_id, flow.name))
        TestBase.logger.info("collected from {}: {}".format(__file__.split('/')[-1], flow.flow_id))

        downloaded_flow = openml.flows.get_flow(flow.flow_id, reinstantiate=True)
        self.assertIsInstance(downloaded_flow.model, sklearn.ensemble.RandomForestClassifier)

    def test_get_flow_reinstantiate_model_no_extension(self):
        # Flow 10 is a WEKA flow
        self.assertRaisesRegex(RuntimeError,
                               "No extension could be found for flow 10: weka.SMO",
                               openml.flows.get_flow,
                               flow_id=10,
                               reinstantiate=True)

    @unittest.skipIf(LooseVersion(sklearn.__version__) == "0.19.1",
                     reason="Target flow is from sklearn 0.19.1")
    def test_get_flow_reinstantiate_model_wrong_version(self):
        # Note that CI does not test against 0.19.1.
        openml.config.server = self.production_server
        _, sklearn_major, _ = LooseVersion(sklearn.__version__).version[:3]
        flow = 8175
        expected = ('Trying to deserialize a model with dependency'
                    ' sklearn==0.19.1 not satisfied.')
        self.assertRaisesRegex(ValueError,
                               expected,
                               openml.flows.get_flow,
                               flow_id=flow,
                               reinstantiate=True)
        if LooseVersion(sklearn.__version__) > "0.19.1":
            # 0.18 actually can't deserialize this because of incompatibility
            flow = openml.flows.get_flow(flow_id=flow, reinstantiate=True,
                                         strict_version=False)
            # ensure that a new flow was created
            assert flow.flow_id is None
            assert "0.19.1" not in flow.dependencies

    def test_get_flow_id(self):
        clf = sklearn.tree.DecisionTreeClassifier()
        flow = openml.extensions.get_extension_by_model(clf).model_to_flow(clf).publish()

        self.assertEqual(openml.flows.get_flow_id(model=clf, exact_version=True), flow.flow_id)
        flow_ids = openml.flows.get_flow_id(model=clf, exact_version=False)
        self.assertIn(flow.flow_id, flow_ids)
        self.assertGreater(len(flow_ids), 2)

        # Check that the output of get_flow_id is identical if only the name is given, no matter
        # whether exact_version is set to True or False.
        flow_ids_exact_version_True = openml.flows.get_flow_id(name=flow.name, exact_version=True)
        flow_ids_exact_version_False = openml.flows.get_flow_id(
            name=flow.name,
            exact_version=False,
        )
        self.assertEqual(flow_ids_exact_version_True, flow_ids_exact_version_False)
        self.assertIn(flow.flow_id, flow_ids_exact_version_True)
        self.assertGreater(len(flow_ids_exact_version_True), 2)
