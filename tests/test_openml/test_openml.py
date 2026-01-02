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

    def test_publish_with_openml_object_merges_tags_and_name(self):
        class Dummy(openml.base.OpenMLBase):
            def __init__(self) -> None:
                self.tags = ["a"]
                self.name = "orig"
                self.published = False

            @property
            def id(self):
                return None

            def _get_repr_body_fields(self):
                return []

            def _to_dict(self):
                return {}

            def _parse_publish_response(self, xml_response):
                return None

            def publish(self):
                self.published = True
                return self

        obj = Dummy()
        result = openml.publish(obj, name="new", tags=["b", "a"])
        assert result is obj
        assert obj.published is True
        assert obj.name == "new"
        assert obj.tags == ["a", "b"]  # dedup and preserve order from original

    @mock.patch("openml.extensions.functions.get_extension_by_model")
    def test_publish_with_extension(self, get_ext_mock):
        flow_mock = mock.MagicMock()
        flow_mock.tags = []
        flow_mock.publish.return_value = "flow-id"

        ext_instance = mock.MagicMock()
        ext_instance.model_to_flow.return_value = flow_mock
        get_ext_mock.return_value = ext_instance

        model = object()
        flow_id = openml.publish(model, name="n", tags=["x"])

        get_ext_mock.assert_called_once_with(model, raise_if_no_extension=True)
        ext_instance.model_to_flow.assert_called_once_with(model)
        assert flow_mock.name == "n"
        assert flow_mock.tags == ["x"]
        flow_mock.publish.assert_called_once_with()
        assert flow_id == "flow-id"
