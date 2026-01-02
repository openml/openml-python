# License: BSD 3-Clause
from __future__ import annotations

from unittest import mock

import openml
from openml.testing import TestBase


class TestInit(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take less
    # than 1 seconds

    @mock.patch("openml.tasks.functions.get_task")
    @mock.patch("openml.datasets.functions.get_dataset")
    @mock.patch("openml.flows.functions.get_flow")
    @mock.patch("openml.runs.functions.get_run")
    def test_populate_cache(
        self,
        run_mock,
        flow_mock,
        dataset_mock,
        task_mock,
    ):
        openml.populate_cache(task_ids=[1, 2], dataset_ids=[3, 4], flow_ids=[5, 6], run_ids=[7, 8])
        assert run_mock.call_count == 2
        for argument, fixture in zip(run_mock.call_args_list, [(7,), (8,)]):
            assert argument[0] == fixture

        assert flow_mock.call_count == 2
        for argument, fixture in zip(flow_mock.call_args_list, [(5,), (6,)]):
            assert argument[0] == fixture

        assert dataset_mock.call_count == 2
        for argument, fixture in zip(
            dataset_mock.call_args_list,
            [(3,), (4,)],
        ):
            assert argument[0] == fixture

        assert task_mock.call_count == 2
        for argument, fixture in zip(task_mock.call_args_list, [(1,), (2,)]):
            assert argument[0] == fixture

    def test_top_level_getters_aliases(self):
        # Ensure top-level convenience aliases point to existing implementations.
        assert openml.list_datasets is openml.datasets.functions.list_datasets
        assert openml.get_dataset is openml.datasets.functions.get_dataset

        assert openml.list_flows is openml.flows.functions.list_flows
        assert openml.get_flow is openml.flows.functions.get_flow

        assert openml.list_runs is openml.runs.functions.list_runs
        assert openml.get_run is openml.runs.functions.get_run

        assert openml.list_tasks is openml.tasks.functions.list_tasks
        assert openml.get_task is openml.tasks.functions.get_task
