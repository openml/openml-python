# License: BSD 3-Clause
from __future__ import annotations

from unittest import mock

import pytest

import openml
from openml.base import OpenMLBase
from openml.testing import TestBase


class DummyOpenMLObject(OpenMLBase):
    def __init__(self, tags: list[str] | None = None, name: str = "orig") -> None:
        self.tags = tags or []
        self.name = name

    @property
    def id(self):
        return None

    def _get_repr_body_fields(self):
        return [("name", self.name)]

    def _to_dict(self):
        return {"oml:dummy": {}}

    def _parse_publish_response(self, xml_response):
        return None


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

    @mock.patch("openml.base.OpenMLBase.publish", autospec=True)
    def test_openml_publish(self, publish_mock):
        obj = DummyOpenMLObject(tags=["a"])
        publish_mock.return_value = obj

        result = openml.publish(obj, name="new", tags=["b", "a"])

        publish_mock.assert_called_once()
        assert publish_mock.call_args[0][0] is obj
        assert result is obj
        assert obj.name == "new"
        assert obj.tags == ["a", "b"]

    @pytest.mark.sklearn()
    @mock.patch("openml.flows.flow.OpenMLFlow.publish", autospec=True)
    def test_openml_publish_ext(self, publish_mock):
        from sklearn.dummy import DummyClassifier

        publish_mock.return_value = mock.sentinel.published

        result = openml.publish(DummyClassifier(), name="n", tags=["x"])

        publish_mock.assert_called_once()
        published_obj = publish_mock.call_args[0][0]
        assert isinstance(published_obj, OpenMLBase)
        assert published_obj.name == "n"
        assert "x" in published_obj.tags
        assert result is mock.sentinel.published
