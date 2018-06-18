import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import six

from openml.testing import TestBase
import openml


class TestInit(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take less
    # than 1 seconds

    @mock.patch('openml.tasks.functions.get_task')
    @mock.patch('openml.datasets.functions.get_dataset')
    @mock.patch('openml.flows.functions.get_flow')
    @mock.patch('openml.runs.functions.get_run')
    def test_populate_cache(self, run_mock, flow_mock, dataset_mock, task_mock):
        openml.populate_cache(task_ids=[1, 2], dataset_ids=[3, 4],
                              flow_ids=[5, 6], run_ids=[7, 8])
        self.assertEqual(run_mock.call_count, 2)
        for argument, fixture in six.moves.zip(run_mock.call_args_list, [(7,), (8,)]):
            self.assertEqual(argument[0], fixture)

        self.assertEqual(flow_mock.call_count, 2)
        for argument, fixture in six.moves.zip(flow_mock.call_args_list, [(5,), (6,)]):
            self.assertEqual(argument[0], fixture)

        self.assertEqual(dataset_mock.call_count, 2)
        for argument, fixture in six.moves.zip(dataset_mock.call_args_list, [(3,), (4,)]):
            self.assertEqual(argument[0], fixture)

        self.assertEqual(task_mock.call_count, 2)
        for argument, fixture in six.moves.zip(task_mock.call_args_list, [(1,), (2,)]):
            self.assertEqual(argument[0], fixture)
            