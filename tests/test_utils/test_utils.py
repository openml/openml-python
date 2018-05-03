from openml.testing import TestBase
import openml


class OpenMLTaskTest(TestBase):
    _multiprocess_can_split_ = True
    _batch_size = 25

    def test_list_all(self):
        required_size = 127  # default test server reset value
        datasets = openml.utils._list_all(openml.datasets.functions._list_datasets,
                                          batch_size=self._batch_size, size=required_size)

        self.assertEquals(len(datasets), required_size)
        for did in datasets:
            self._check_dataset(datasets[did])

    def test_list_all_for_datasets(self):
        required_size = 127  # default test server reset value
        datasets = openml.datasets.list_datasets(batch_size=self._batch_size, size=required_size)

        self.assertEquals(len(datasets), required_size)
        for did in datasets:
            self._check_dataset(datasets[did])

    def test_list_all_for_tasks(self):
        required_size = 1068  # default test server reset value
        tasks = openml.tasks.list_tasks(batch_size=self._batch_size, size=required_size)

        self.assertEquals(len(tasks), required_size)

    def test_list_all_for_flows(self):
        required_size = 15  # default test server reset value
        flows = openml.flows.list_flows(batch_size=self._batch_size, size=required_size)

        self.assertEquals(len(flows), required_size)

    def test_list_all_for_setups(self):
        required_size = 50
        # TODO apparently list_setups function does not support kwargs
        setups = openml.setups.list_setups(size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEquals(len(setups), required_size)

    def test_list_all_for_runs(self):
        required_size = 48
        runs = openml.runs.list_runs(batch_size=self._batch_size, size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEquals(len(runs), required_size)

    def test_list_all_for_evaluations(self):
        required_size = 57
        # TODO apparently list_evaluations function does not support kwargs
        evaluations = openml.evaluations.list_evaluations(function='predictive_accuracy',
                                                          size=required_size)

        # might not be on test server after reset, please rerun test at least once if fails
        self.assertEquals(len(evaluations), required_size)
