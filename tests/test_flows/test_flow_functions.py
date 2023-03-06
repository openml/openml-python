# License: BSD 3-Clause

from collections import OrderedDict
import copy
import functools
import unittest
from unittest import mock
from unittest.mock import patch

from distutils.version import LooseVersion

import requests
import sklearn
from sklearn import ensemble
import pandas as pd
import pytest

import openml
from openml.exceptions import OpenMLNotAuthorizedError, OpenMLServerException
from openml.testing import TestBase
import openml.extensions.sklearn


@pytest.mark.usefixtures("long_version")
class TestFlowFunctions(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super(TestFlowFunctions, self).setUp()

    def tearDown(self):
        super(TestFlowFunctions, self).tearDown()

    def _check_flow(self, flow):
        self.assertEqual(type(flow), dict)
        self.assertEqual(len(flow), 6)
        self.assertIsInstance(flow["id"], int)
        self.assertIsInstance(flow["name"], str)
        self.assertIsInstance(flow["full_name"], str)
        self.assertIsInstance(flow["version"], str)
        # There are some runs on openml.org that can have an empty external version
        ext_version_str_or_none = (
            isinstance(flow["external_version"], str) or flow["external_version"] is None
        )
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

    def test_list_flows_output_format(self):
        openml.config.server = self.production_server
        # We can only perform a smoke test here because we test on dynamic
        # data from the internet...
        flows = openml.flows.list_flows(output_format="dataframe")
        self.assertIsInstance(flows, pd.DataFrame)
        self.assertGreaterEqual(len(flows), 1500)

    def test_list_flows_empty(self):
        openml.config.server = self.production_server
        flows = openml.flows.list_flows(tag="NoOneEverUsesThisTag123")
        if len(flows) > 0:
            raise ValueError("UnitTest Outdated, got somehow results (please adapt)")

        self.assertIsInstance(flows, dict)

    def test_list_flows_by_tag(self):
        openml.config.server = self.production_server
        flows = openml.flows.list_flows(tag="weka")
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
        flow = openml.flows.OpenMLFlow(
            name="Test",
            description="Test flow",
            model=None,
            components=OrderedDict(),
            parameters=OrderedDict(),
            parameters_meta_info=OrderedDict(),
            external_version="1",
            tags=["abc", "def"],
            language="English",
            dependencies="abc",
            class_name="Test",
            custom_name="Test",
        )

        # Test most important values that can be set by a user
        openml.flows.functions.assert_flows_equal(flow, flow)
        for attribute, new_value in [
            ("name", "Tes"),
            ("external_version", "2"),
            ("language", "english"),
            ("dependencies", "ab"),
            ("class_name", "Tes"),
            ("custom_name", "Tes"),
        ]:
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
        for attribute, new_value in [
            ("flow_id", 1),
            ("uploader", 1),
            ("version", 1),
            ("upload_date", "18.12.1988"),
            ("binary_url", "openml.org"),
            ("binary_format", "gzip"),
            ("binary_md5", "12345"),
            ("model", []),
            ("tags", ["abc", "de"]),
        ]:
            new_flow = copy.deepcopy(flow)
            setattr(new_flow, attribute, new_value)
            self.assertNotEqual(
                getattr(flow, attribute),
                getattr(new_flow, attribute),
            )
            openml.flows.functions.assert_flows_equal(flow, new_flow)

        # Now test for parameters
        flow.parameters["abc"] = 1.0
        flow.parameters["def"] = 2.0
        openml.flows.functions.assert_flows_equal(flow, flow)
        new_flow = copy.deepcopy(flow)
        new_flow.parameters["abc"] = 3.0
        self.assertRaises(ValueError, openml.flows.functions.assert_flows_equal, flow, new_flow)

        # Now test for components (subflows)
        parent_flow = copy.deepcopy(flow)
        subflow = copy.deepcopy(flow)
        parent_flow.components["subflow"] = subflow
        openml.flows.functions.assert_flows_equal(parent_flow, parent_flow)
        self.assertRaises(
            ValueError, openml.flows.functions.assert_flows_equal, parent_flow, subflow
        )
        new_flow = copy.deepcopy(parent_flow)
        new_flow.components["subflow"].name = "Subflow name"
        self.assertRaises(
            ValueError, openml.flows.functions.assert_flows_equal, parent_flow, new_flow
        )

    def test_are_flows_equal_ignore_parameter_values(self):
        paramaters = OrderedDict((("a", 5), ("b", 6)))
        parameters_meta_info = OrderedDict((("a", None), ("b", None)))

        flow = openml.flows.OpenMLFlow(
            name="Test",
            description="Test flow",
            model=None,
            components=OrderedDict(),
            parameters=paramaters,
            parameters_meta_info=parameters_meta_info,
            external_version="1",
            tags=["abc", "def"],
            language="English",
            dependencies="abc",
            class_name="Test",
            custom_name="Test",
        )

        openml.flows.functions.assert_flows_equal(flow, flow)
        openml.flows.functions.assert_flows_equal(flow, flow, ignore_parameter_values=True)

        new_flow = copy.deepcopy(flow)
        new_flow.parameters["a"] = 7
        self.assertRaisesRegex(
            ValueError,
            r"values for attribute 'parameters' differ: "
            r"'OrderedDict\(\[\('a', 5\), \('b', 6\)\]\)'\nvs\n"
            r"'OrderedDict\(\[\('a', 7\), \('b', 6\)\]\)'",
            openml.flows.functions.assert_flows_equal,
            flow,
            new_flow,
        )
        openml.flows.functions.assert_flows_equal(flow, new_flow, ignore_parameter_values=True)

        del new_flow.parameters["a"]
        self.assertRaisesRegex(
            ValueError,
            r"values for attribute 'parameters' differ: "
            r"'OrderedDict\(\[\('a', 5\), \('b', 6\)\]\)'\nvs\n"
            r"'OrderedDict\(\[\('b', 6\)\]\)'",
            openml.flows.functions.assert_flows_equal,
            flow,
            new_flow,
        )
        self.assertRaisesRegex(
            ValueError,
            r"Flow Test: parameter set of flow differs from the parameters "
            r"stored on the server.",
            openml.flows.functions.assert_flows_equal,
            flow,
            new_flow,
            ignore_parameter_values=True,
        )

    def test_are_flows_equal_ignore_if_older(self):
        paramaters = OrderedDict((("a", 5), ("b", 6)))
        parameters_meta_info = OrderedDict((("a", None), ("b", None)))
        flow_upload_date = "2017-01-31T12-01-01"
        assert_flows_equal = openml.flows.functions.assert_flows_equal

        flow = openml.flows.OpenMLFlow(
            name="Test",
            description="Test flow",
            model=None,
            components=OrderedDict(),
            parameters=paramaters,
            parameters_meta_info=parameters_meta_info,
            external_version="1",
            tags=["abc", "def"],
            language="English",
            dependencies="abc",
            class_name="Test",
            custom_name="Test",
            upload_date=flow_upload_date,
        )

        assert_flows_equal(flow, flow, ignore_parameter_values_on_older_children=flow_upload_date)
        assert_flows_equal(flow, flow, ignore_parameter_values_on_older_children=None)
        new_flow = copy.deepcopy(flow)
        new_flow.parameters["a"] = 7
        self.assertRaises(
            ValueError,
            assert_flows_equal,
            flow,
            new_flow,
            ignore_parameter_values_on_older_children=flow_upload_date,
        )
        self.assertRaises(
            ValueError,
            assert_flows_equal,
            flow,
            new_flow,
            ignore_parameter_values_on_older_children=None,
        )

        new_flow.upload_date = "2016-01-31T12-01-01"
        self.assertRaises(
            ValueError,
            assert_flows_equal,
            flow,
            new_flow,
            ignore_parameter_values_on_older_children=flow_upload_date,
        )
        assert_flows_equal(flow, flow, ignore_parameter_values_on_older_children=None)

    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="OrdinalEncoder introduced in 0.20. "
        "No known models with list of lists parameters in older versions.",
    )
    def test_sklearn_to_flow_list_of_lists(self):
        from sklearn.preprocessing import OrdinalEncoder

        ordinal_encoder = OrdinalEncoder(categories=[[0, 1], [0, 1]])
        extension = openml.extensions.sklearn.SklearnExtension()

        # Test serialization works
        flow = extension.model_to_flow(ordinal_encoder)

        # Test flow is accepted by server
        self._add_sentinel_to_flow_name(flow)
        flow.publish()
        TestBase._mark_entity_for_removal("flow", (flow.flow_id, flow.name))
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))
        # Test deserialization works
        server_flow = openml.flows.get_flow(flow.flow_id, reinstantiate=True)
        self.assertEqual(server_flow.parameters["categories"], "[[0, 1], [0, 1]]")
        self.assertEqual(server_flow.model.categories, flow.model.categories)

    def test_get_flow1(self):
        # Regression test for issue #305
        # Basically, this checks that a flow without an external version can be loaded
        openml.config.server = self.production_server
        flow = openml.flows.get_flow(1)
        self.assertIsNone(flow.external_version)

    def test_get_flow_reinstantiate_model(self):
        model = ensemble.RandomForestClassifier(n_estimators=33)
        extension = openml.extensions.get_extension_by_model(model)
        flow = extension.model_to_flow(model)
        flow.publish(raise_error_if_exists=False)
        TestBase._mark_entity_for_removal("flow", (flow.flow_id, flow.name))
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))

        downloaded_flow = openml.flows.get_flow(flow.flow_id, reinstantiate=True)
        self.assertIsInstance(downloaded_flow.model, sklearn.ensemble.RandomForestClassifier)

    def test_get_flow_reinstantiate_model_no_extension(self):
        # Flow 10 is a WEKA flow
        self.assertRaisesRegex(
            RuntimeError,
            "No extension could be found for flow 10: weka.SMO",
            openml.flows.get_flow,
            flow_id=10,
            reinstantiate=True,
        )

    @unittest.skipIf(
        LooseVersion(sklearn.__version__) == "0.19.1",
        reason="Requires scikit-learn!=0.19.1, because target flow is from that version.",
    )
    def test_get_flow_with_reinstantiate_strict_with_wrong_version_raises_exception(self):
        openml.config.server = self.production_server
        flow = 8175
        expected = "Trying to deserialize a model with dependency sklearn==0.19.1 not satisfied."
        self.assertRaisesRegex(
            ValueError,
            expected,
            openml.flows.get_flow,
            flow_id=flow,
            reinstantiate=True,
            strict_version=True,
        )

    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "1" and LooseVersion(sklearn.__version__) != "1.0.0",
        reason="Requires scikit-learn < 1.0.1."
        # Because scikit-learn dropped min_impurity_split hyperparameter in 1.0,
        # and the requested flow is from 1.0.0 exactly.
    )
    def test_get_flow_reinstantiate_flow_not_strict_post_1(self):
        openml.config.server = self.production_server
        flow = openml.flows.get_flow(flow_id=19190, reinstantiate=True, strict_version=False)
        assert flow.flow_id is None
        assert "sklearn==1.0.0" not in flow.dependencies

    @unittest.skipIf(
        (LooseVersion(sklearn.__version__) < "0.23.2")
        or ("1.0" < LooseVersion(sklearn.__version__)),
        reason="Requires scikit-learn 0.23.2 or ~0.24."
        # Because these still have min_impurity_split, but with new scikit-learn module structure."
    )
    def test_get_flow_reinstantiate_flow_not_strict_023_and_024(self):
        openml.config.server = self.production_server
        flow = openml.flows.get_flow(flow_id=18587, reinstantiate=True, strict_version=False)
        assert flow.flow_id is None
        assert "sklearn==0.23.1" not in flow.dependencies

    @unittest.skipIf(
        "0.23" < LooseVersion(sklearn.__version__),
        reason="Requires scikit-learn<=0.23, because the scikit-learn module structure changed.",
    )
    def test_get_flow_reinstantiate_flow_not_strict_pre_023(self):
        openml.config.server = self.production_server
        flow = openml.flows.get_flow(flow_id=8175, reinstantiate=True, strict_version=False)
        assert flow.flow_id is None
        assert "sklearn==0.19.1" not in flow.dependencies

    def test_get_flow_id(self):
        if self.long_version:
            list_all = openml.utils._list_all
        else:
            list_all = functools.lru_cache()(openml.utils._list_all)
        with patch("openml.utils._list_all", list_all):
            clf = sklearn.tree.DecisionTreeClassifier()
            flow = openml.extensions.get_extension_by_model(clf).model_to_flow(clf).publish()
            TestBase._mark_entity_for_removal("flow", (flow.flow_id, flow.name))
            TestBase.logger.info(
                "collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id)
            )

            self.assertEqual(openml.flows.get_flow_id(model=clf, exact_version=True), flow.flow_id)
            flow_ids = openml.flows.get_flow_id(model=clf, exact_version=False)
            self.assertIn(flow.flow_id, flow_ids)
            self.assertGreater(len(flow_ids), 0)

            # Check that the output of get_flow_id is identical if only the name is given, no matter
            # whether exact_version is set to True or False.
            flow_ids_exact_version_True = openml.flows.get_flow_id(
                name=flow.name, exact_version=True
            )
            flow_ids_exact_version_False = openml.flows.get_flow_id(
                name=flow.name,
                exact_version=False,
            )
            self.assertEqual(flow_ids_exact_version_True, flow_ids_exact_version_False)
            self.assertIn(flow.flow_id, flow_ids_exact_version_True)

    def test_delete_flow(self):
        flow = openml.OpenMLFlow(
            name="sklearn.dummy.DummyClassifier",
            class_name="sklearn.dummy.DummyClassifier",
            description="test description",
            model=sklearn.dummy.DummyClassifier(),
            components=OrderedDict(),
            parameters=OrderedDict(),
            parameters_meta_info=OrderedDict(),
            external_version="1",
            tags=[],
            language="English",
            dependencies=None,
        )

        flow, _ = self._add_sentinel_to_flow_name(flow, None)

        flow.publish()
        _flow_id = flow.flow_id
        self.assertTrue(openml.flows.delete_flow(_flow_id))


@mock.patch.object(requests.Session, "delete")
def test_delete_flow_not_owned(mock_get, test_files_directory):
    openml.config.start_using_configuration_for_example()
    with open(
        test_files_directory / "mock_responses" / "flows" / "flow_delete_not_owned.xml", "r"
    ) as xml_response:
        response_body = xml_response.read()

    response = requests.Response()
    response.status_code = 323
    response._content = response_body.encode()
    mock_get.return_value = response

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The flow can not be deleted because it was not uploaded by you.",
    ):
        openml.flows.delete_flow(40_000)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/flow/40000",),
        {"params": {"api_key": "c0c42819af31e706efe1f4b88c23c6c1"}},
    ]
    assert expected_call_args == list(mock_get.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_flow_with_run(mock_get, test_files_directory):
    openml.config.start_using_configuration_for_example()
    with open(
        test_files_directory / "mock_responses" / "flows" / "flow_delete_has_runs.xml", "r"
    ) as xml_response:
        response_body = xml_response.read()

    response = requests.Response()
    response.status_code = 324
    response._content = response_body.encode()
    mock_get.return_value = response

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The flow can not be deleted because it still has associated entities:",
    ):
        openml.flows.delete_flow(40_000)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/flow/40000",),
        {"params": {"api_key": "c0c42819af31e706efe1f4b88c23c6c1"}},
    ]
    assert expected_call_args == list(mock_get.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_subflow(mock_get, test_files_directory):
    openml.config.start_using_configuration_for_example()
    with open(
        test_files_directory / "mock_responses" / "flows" / "flow_delete_is_subflow.xml", "r"
    ) as xml_response:
        response_body = xml_response.read()

    response = requests.Response()
    response.status_code = 328
    response._content = response_body.encode()
    mock_get.return_value = response

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The flow can not be deleted because it still has associated entities:",
    ):
        openml.flows.delete_flow(40_000)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/flow/40000",),
        {"params": {"api_key": "c0c42819af31e706efe1f4b88c23c6c1"}},
    ]
    assert expected_call_args == list(mock_get.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_flow_success(mock_get, test_files_directory):
    openml.config.start_using_configuration_for_example()
    with open(
        test_files_directory / "mock_responses" / "flows" / "flow_delete_successful.xml", "r"
    ) as xml_response:
        response_body = xml_response.read()

    response = requests.Response()
    response.status_code = 200
    response._content = response_body.encode()
    mock_get.return_value = response

    success = openml.flows.delete_flow(33364)
    assert success

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/flow/33364",),
        {"params": {"api_key": "c0c42819af31e706efe1f4b88c23c6c1"}},
    ]
    assert expected_call_args == list(mock_get.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_unknown_flow(mock_get, test_files_directory):
    openml.config.start_using_configuration_for_example()
    with open(
        test_files_directory / "mock_responses" / "flows" / "flow_delete_not_exist.xml", "r"
    ) as xml_response:
        response_body = xml_response.read()

    response = requests.Response()
    response.status_code = 322
    response._content = response_body.encode()
    mock_get.return_value = response

    with pytest.raises(
        OpenMLServerException,
        match="flow does not exist",
    ):
        openml.flows.delete_flow(9_999_999)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/flow/9999999",),
        {"params": {"api_key": "c0c42819af31e706efe1f4b88c23c6c1"}},
    ]
    assert expected_call_args == list(mock_get.call_args)
