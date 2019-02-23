from collections import OrderedDict
import copy

import openml
from openml.testing import TestBase


class TestFlowFunctions(TestBase):
    _multiprocess_can_split_ = True

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
                                     ('description', 'Test flo'),
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
