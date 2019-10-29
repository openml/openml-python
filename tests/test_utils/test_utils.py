from openml.testing import TestBase
import numpy as np
import openml
import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock


class OpenMLTaskTest(TestBase):
    _multiprocess_can_split_ = True
    _batch_size = 25

    def mocked_perform_api_call(call, request_method):
        # TODO: JvR: Why is this not a staticmethod?
        url = openml.config.server + '/' + call
        return openml._api_calls._read_url(url, request_method=request_method)

    def test_list_all(self):
        openml.utils._list_all(listing_call=openml.tasks.functions._list_tasks)

    @mock.patch('openml._api_calls._perform_api_call',
                side_effect=mocked_perform_api_call)
    def test_list_all_few_results_available(self, _perform_api_call):
        # we want to make sure that the number of api calls is only 1.
        # Although we have multiple versions of the iris dataset, there is only
        # one with this name/version combination

        datasets = openml.datasets.list_datasets(size=1000,
                                                 data_name='iris',
                                                 data_version=1)
        self.assertEqual(len(datasets), 1)
        self.assertEqual(_perform_api_call.call_count, 1)

    def test_list_all_for_datasets(self):
        required_size = 127  # default test server reset value
        datasets = openml.datasets.list_datasets(batch_size=self._batch_size, size=required_size)

        self.assertEqual(len(datasets), required_size)
        for did in datasets:
            self._check_dataset(datasets[did])

    def test_list_datasets_with_high_size_parameter(self):
        # Testing on prod since concurrent deletion of uploded datasets make the test fail
        openml.config.server = self.production_server

        datasets_a = openml.datasets.list_datasets()
        datasets_b = openml.datasets.list_datasets(size=np.inf)

        # Reverting to test server
        openml.config.server = self.test_server

        self.assertEqual(len(datasets_a), len(datasets_b))

    def test_list_all_for_tasks(self):
        required_size = 1068  # default test server reset value
        tasks = openml.tasks.list_tasks(batch_size=self._batch_size, size=required_size)

        self.assertEqual(len(tasks), required_size)

    def test_list_all_for_flows(self):
        required_size = 15  # default test server reset value
        flows = openml.flows.list_flows(batch_size=self._batch_size, size=required_size)

        self.assertEqual(len(flows), required_size)

    def test_list_all_for_setups(self):
        required_size = 50
        # TODO apparently list_setups function does not support kwargs
        setups = openml.setups.list_setups(size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEqual(len(setups), required_size)

    def test_list_all_for_runs(self):
        required_size = 21
        runs = openml.runs.list_runs(batch_size=self._batch_size, size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEqual(len(runs), required_size)

    def test_list_all_for_evaluations(self):
        required_size = 22
        # TODO apparently list_evaluations function does not support kwargs
        evaluations = openml.evaluations.list_evaluations(function='predictive_accuracy',
                                                          size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEqual(len(evaluations), required_size)
