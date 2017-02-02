import unittest

import openml
from openml.util import is_string


class TestFlowFunctions(unittest.TestCase):
    def _check_flow(self, flow):
        self.assertEqual(type(flow), dict)
        self.assertEqual(len(flow), 6)
        self.assertIsInstance(flow['id'], int)
        self.assertTrue(is_string(flow['name']))
        self.assertTrue(is_string(flow['full_name']))
        self.assertTrue(is_string(flow['version']))
        # There are some runs on openml.org that can have an empty external
        # version
        self.assertTrue(is_string(flow['external_version']) or
                        flow['external_version'] is None)

    def test_list_flows(self):
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        flows = openml.flows.list_flows()
        # 3000 as the number of flows on openml.org
        self.assertGreaterEqual(len(flows), 3000)
        for fid in flows:
            self._check_flow(flows[fid])

    def test_list_datasets_by_tag(self):
        flows = openml.flows.list_flows(tag='weka')
        self.assertGreaterEqual(len(flows), 5)
        for did in flows:
            self._check_flow(flows[did])

    def test_list_datasets_paginate(self):
        size = 10
        max = 100
        for i in range(0, max, size):
            flows = openml.flows.list_flows(offset=i, size=size)
            self.assertGreaterEqual(size, len(flows))
            for did in flows:
                self._check_flow(flows[did])