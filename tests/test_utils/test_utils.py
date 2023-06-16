import os
import tempfile
import unittest.mock

import openml
from openml.testing import TestBase


class OpenMLTaskTest(TestBase):
    _multiprocess_can_split_ = True

    def mocked_perform_api_call(call, request_method):
        # TODO: JvR: Why is this not a staticmethod?
        url = openml.config.server + "/" + call
        return openml._api_calls._download_text_file(url)

    def test_list_all(self):
        openml.utils._list_all(listing_call=openml.tasks.functions._list_tasks)
        openml.utils._list_all(
            listing_call=openml.tasks.functions._list_tasks, output_format="dataframe"
        )

    def test_list_all_with_multiple_batches(self):
        res = openml.utils._list_all(
            listing_call=openml.tasks.functions._list_tasks, output_format="dict", batch_size=1050
        )
        # Verify that test server state is still valid for this test to work as intended
        #  -> If the number of results is less than 1050, the test can not test the
        #  batching operation. By having more than 1050 results we know that batching
        # was triggered. 1050 appears to be a number of tasks that is available on a fresh
        # test server.
        assert len(res) > 1050
        res2 = openml.utils._list_all(
            listing_call=openml.tasks.functions._list_tasks,
            output_format="dataframe",
            batch_size=1050,
        )
        # It might be possible that someone added a new task in the meantime
        assert len(res) <= len(res2)

    @unittest.mock.patch("openml._api_calls._perform_api_call", side_effect=mocked_perform_api_call)
    def test_list_all_few_results_available(self, _perform_api_call):
        # we want to make sure that the number of api calls is only 1.
        # Although we have multiple versions of the iris dataset, there is only
        # one with this name/version combination

        datasets = openml.datasets.list_datasets(
            size=1000, data_name="iris", data_version=1, output_format="dataframe"
        )
        self.assertEqual(len(datasets), 1)
        self.assertEqual(_perform_api_call.call_count, 1)

    def test_list_all_for_datasets(self):
        required_size = 127  # default test server reset value
        datasets = openml.datasets.list_datasets(
            batch_size=100, size=required_size, output_format="dataframe"
        )

        self.assertEqual(len(datasets), required_size)
        for dataset in datasets.to_dict(orient="index").values():
            self._check_dataset(dataset)

    def test_list_all_for_tasks(self):
        required_size = 1068  # default test server reset value
        tasks = openml.tasks.list_tasks(
            batch_size=1000, size=required_size, output_format="dataframe"
        )
        self.assertEqual(len(tasks), required_size)

    def test_list_all_for_flows(self):
        required_size = 15  # default test server reset value
        flows = openml.flows.list_flows(
            batch_size=25, size=required_size, output_format="dataframe"
        )
        self.assertEqual(len(flows), required_size)

    def test_list_all_for_setups(self):
        required_size = 50
        # TODO apparently list_setups function does not support kwargs
        setups = openml.setups.list_setups(size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEqual(len(setups), required_size)

    def test_list_all_for_runs(self):
        required_size = 21
        runs = openml.runs.list_runs(batch_size=25, size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEqual(len(runs), required_size)

    def test_list_all_for_evaluations(self):
        required_size = 22
        # TODO apparently list_evaluations function does not support kwargs
        evaluations = openml.evaluations.list_evaluations(
            function="predictive_accuracy", size=required_size
        )

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEqual(len(evaluations), required_size)

    @unittest.mock.patch("openml.config.get_cache_directory")
    @unittest.skipIf(os.name == "nt", "https://github.com/openml/openml-python/issues/1033")
    def test__create_cache_directory(self, config_mock):
        with tempfile.TemporaryDirectory(dir=self.workdir) as td:
            config_mock.return_value = td
            openml.utils._create_cache_directory("abc")
            self.assertTrue(os.path.exists(os.path.join(td, "abc")))
            subdir = os.path.join(td, "def")
            os.mkdir(subdir)
            os.chmod(subdir, 0o444)
            config_mock.return_value = subdir
            with self.assertRaisesRegex(
                openml.exceptions.OpenMLCacheException,
                r"Cannot create cache directory",
            ):
                openml.utils._create_cache_directory("ghi")
