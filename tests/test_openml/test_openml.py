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

    @mock.patch("openml.tasks.functions.list_tasks")
    @mock.patch("openml.datasets.functions.list_datasets")
    def test_list_dispatch(self, list_datasets_mock, list_tasks_mock):
        # Need to patch after import, so update dispatch dict
        with mock.patch.dict(
            "openml.dispatchers._LIST_DISPATCH",
            {
                "dataset": list_datasets_mock,
                "task": list_tasks_mock,
            },
        ):
            openml.list_all("dataset")
            list_datasets_mock.assert_called_once_with()

            openml.list_all("task", size=5)
            list_tasks_mock.assert_called_once_with(size=5)

    @mock.patch("openml.tasks.functions.get_task")
    @mock.patch("openml.datasets.functions.get_dataset")
    def test_get_dispatch(self, get_dataset_mock, get_task_mock):
        # Need to patch after import, so update dispatch dict
        with mock.patch.dict(
            "openml.dispatchers._GET_DISPATCH",
            {
                "dataset": get_dataset_mock,
                "task": get_task_mock,
            },
        ):
            openml.get(61)
            get_dataset_mock.assert_called_with(61)

            openml.get("Fashion-MNIST", version=2)
            get_dataset_mock.assert_called_with("Fashion-MNIST", version=2)

            openml.get("Fashion-MNIST")
            get_dataset_mock.assert_called_with("Fashion-MNIST")

            openml.get(31, object_type="task")
            get_task_mock.assert_called_with(31)
