# License: BSD 3-Clause
from __future__ import annotations

import collections
import copy
import hashlib
import re
import os
import time
from packaging.version import Version
from unittest import mock

import pytest
import requests
import scipy.stats
import sklearn
import sklearn.datasets
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree
import xmltodict

from openml_sklearn import SklearnExtension

import openml
import openml.exceptions
import openml.utils
from openml._api_calls import _perform_api_call
from openml.testing import SimpleImputer, TestBase, create_request_response


class TestFlow(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()
        self.extension = SklearnExtension()

    def tearDown(self):
        super().tearDown()

    @pytest.mark.sklearn()
    def test_to_xml_from_xml(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        estimator_name = (
            "base_estimator" if Version(sklearn.__version__) < Version("1.4") else "estimator"
        )
        boosting = sklearn.ensemble.AdaBoostClassifier(
            **{estimator_name: sklearn.tree.DecisionTreeClassifier()},
        )
        model = sklearn.pipeline.Pipeline(steps=(("scaler", scaler), ("boosting", boosting)))
        flow = self.extension.model_to_flow(model)
        flow.flow_id = -234
        # end of setup

        xml = flow._to_xml()
        xml_dict = xmltodict.parse(xml)
        new_flow = openml.flows.OpenMLFlow._from_dict(xml_dict)

        # Would raise exception if they are not legal
        openml.flows.functions.assert_flows_equal(new_flow, flow)
        assert new_flow is not flow

    @pytest.mark.sklearn()
    @mock.patch("openml.flows.functions.flow_exists")
    def test_publish_existing_flow(self, flow_exists_mock):
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=2)
        flow = self.extension.model_to_flow(clf)
        flow_exists_mock.return_value = 1

        with pytest.raises(openml.exceptions.PyOpenMLError, match="OpenMLFlow already exists"):
            flow.publish(raise_error_if_exists=True)

        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(
            f"collected from {__file__.split('/')[-1]}: {flow.flow_id}",
        )

    @pytest.mark.sklearn()
    @mock.patch("openml.flows.functions.get_flow")
    @mock.patch("openml.flows.functions.flow_exists")
    @mock.patch("openml._api_calls._perform_api_call")
    def test_publish_error(self, api_call_mock, flow_exists_mock, get_flow_mock):
        model = sklearn.ensemble.RandomForestClassifier()
        flow = self.extension.model_to_flow(model)
        api_call_mock.return_value = (
            "<oml:upload_flow>\n" "    <oml:id>1</oml:id>\n" "</oml:upload_flow>"
        )
        flow_exists_mock.return_value = False
        get_flow_mock.return_value = flow

        flow.publish()
        # Not collecting flow_id for deletion since this is a test for failed upload

        assert api_call_mock.call_count == 1
        assert get_flow_mock.call_count == 1
        assert flow_exists_mock.call_count == 1

        flow_copy = copy.deepcopy(flow)
        flow_copy.name = flow_copy.name[:-1]
        get_flow_mock.return_value = flow_copy
        flow_exists_mock.return_value = 1

        if Version(sklearn.__version__) < Version("0.22"):
            fixture = (
                "The flow on the server is inconsistent with the local flow. "
                "The server flow ID is 1. Please check manually and remove "
                "the flow if necessary! Error is:\n"
                "'Flow sklearn.ensemble.forest.RandomForestClassifier: "
                "values for attribute 'name' differ: "
                "'sklearn.ensemble.forest.RandomForestClassifier'"
                "\nvs\n'sklearn.ensemble.forest.RandomForestClassifie'.'"
            )
        else:
            # sklearn.ensemble.forest -> sklearn.ensemble._forest
            fixture = (
                "The flow on the server is inconsistent with the local flow. "
                "The server flow ID is 1. Please check manually and remove "
                "the flow if necessary! Error is:\n"
                "'Flow sklearn.ensemble._forest.RandomForestClassifier: "
                "values for attribute 'name' differ: "
                "'sklearn.ensemble._forest.RandomForestClassifier'"
                "\nvs\n'sklearn.ensemble._forest.RandomForestClassifie'.'"
            )
        with pytest.raises(ValueError, match=fixture):
            flow.publish()

        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(
            f"collected from {__file__.split('/')[-1]}: {flow.flow_id}",
        )

        assert get_flow_mock.call_count == 2

    @pytest.mark.sklearn()
    def test_illegal_flow(self):
        # should throw error as it contains two imputers
        illegal = sklearn.pipeline.Pipeline(
            steps=[
                ("imputer1", SimpleImputer()),
                ("imputer2", SimpleImputer()),
                ("classif", sklearn.tree.DecisionTreeClassifier()),
            ],
        )
        self.assertRaises(ValueError, self.extension.model_to_flow, illegal)

    def test_extract_tags(self):
        flow_xml = "<oml:tag>study_14</oml:tag>"
        flow_dict = xmltodict.parse(flow_xml)
        tags = openml.utils.extract_xml_tags("oml:tag", flow_dict)
        assert tags == ["study_14"]

        flow_xml = "<oml:flow><oml:tag>OpenmlWeka</oml:tag>\n" "<oml:tag>weka</oml:tag></oml:flow>"
        flow_dict = xmltodict.parse(flow_xml)
        tags = openml.utils.extract_xml_tags("oml:tag", flow_dict["oml:flow"])
        assert tags == ["OpenmlWeka", "weka"]

    @pytest.mark.sklearn()
    @pytest.mark.test_server()
    def test_sklearn_to_upload_to_flow(self):
        iris = sklearn.datasets.load_iris()
        X = iris.data
        y = iris.target

        # Test a more complicated flow
        ohe_params = {"handle_unknown": "ignore"}
        if Version(sklearn.__version__) >= Version("0.20"):
            ohe_params["categories"] = "auto"
        ohe = sklearn.preprocessing.OneHotEncoder(**ohe_params)
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        pca = sklearn.decomposition.TruncatedSVD()
        fs = sklearn.feature_selection.SelectPercentile(
            score_func=sklearn.feature_selection.f_classif,
            percentile=30,
        )
        fu = sklearn.pipeline.FeatureUnion(transformer_list=[("pca", pca), ("fs", fs)])
        estimator_name = (
            "base_estimator" if Version(sklearn.__version__) < Version("1.4") else "estimator"
        )
        boosting = sklearn.ensemble.AdaBoostClassifier(
            **{estimator_name: sklearn.tree.DecisionTreeClassifier()},
        )
        model = sklearn.pipeline.Pipeline(
            steps=[("ohe", ohe), ("scaler", scaler), ("fu", fu), ("boosting", boosting)],
        )
        parameter_grid = {
            "boosting__n_estimators": [1, 5, 10, 100],
            "boosting__learning_rate": scipy.stats.uniform(0.01, 0.99),
            f"boosting__{estimator_name}__max_depth": scipy.stats.randint(1, 10),
        }
        cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)
        rs = sklearn.model_selection.RandomizedSearchCV(
            estimator=model,
            param_distributions=parameter_grid,
            cv=cv,
        )
        rs.fit(X, y)
        flow = self.extension.model_to_flow(rs)
        # Tags may be sorted in any order (by the server). Just using one tag
        # makes sure that the xml comparison does not fail because of that.
        subflows = [flow]
        while len(subflows) > 0:
            f = subflows.pop()
            f.tags = []
            subflows.extend(list(f.components.values()))

        flow, sentinel = self._add_sentinel_to_flow_name(flow, None)

        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")
        assert isinstance(flow.flow_id, int)

        # Check whether we can load the flow again
        # Remove the sentinel from the name again so that we can reinstantiate
        # the object again
        new_flow = openml.flows.get_flow(flow_id=flow.flow_id, reinstantiate=True)

        local_xml = flow._to_xml()
        server_xml = new_flow._to_xml()

        for _i in range(10):
            # Make sure that we replace all occurences of two newlines
            local_xml = local_xml.replace(sentinel, "")
            local_xml = (
                local_xml.replace("  ", "")
                .replace("\t", "")
                .strip()
                .replace("\n\n", "\n")
                .replace("&quot;", '"')
            )
            local_xml = re.sub(r"(^$)", "", local_xml)
            server_xml = server_xml.replace(sentinel, "")
            server_xml = (
                server_xml.replace("  ", "")
                .replace("\t", "")
                .strip()
                .replace("\n\n", "\n")
                .replace("&quot;", '"')
            )
            server_xml = re.sub(r"^$", "", server_xml)

        assert server_xml == local_xml

        # Would raise exception if they are not equal!
        openml.flows.functions.assert_flows_equal(new_flow, flow)
        assert new_flow is not flow

        # OneHotEncoder was moved to _encoders module in 0.20
        module_name_encoder = (
            "_encoders" if Version(sklearn.__version__) >= Version("0.20") else "data"
        )
        if Version(sklearn.__version__) < Version("0.22"):
            fixture_name = (
                f"{sentinel}sklearn.model_selection._search.RandomizedSearchCV("
                "estimator=sklearn.pipeline.Pipeline("
                f"ohe=sklearn.preprocessing.{module_name_encoder}.OneHotEncoder,"
                "scaler=sklearn.preprocessing.data.StandardScaler,"
                "fu=sklearn.pipeline.FeatureUnion("
                "pca=sklearn.decomposition.truncated_svd.TruncatedSVD,"
                "fs="
                "sklearn.feature_selection.univariate_selection.SelectPercentile),"
                "boosting=sklearn.ensemble.weight_boosting.AdaBoostClassifier("
                "base_estimator=sklearn.tree.tree.DecisionTreeClassifier)))"
            )
        else:
            # sklearn.sklearn.preprocessing.data -> sklearn.sklearn.preprocessing._data
            # sklearn.sklearn.decomposition.truncated_svd -> sklearn.decomposition._truncated_svd
            # sklearn.feature_selection.univariate_selection ->
            #     sklearn.feature_selection._univariate_selection
            # sklearn.ensemble.weight_boosting -> sklearn.ensemble._weight_boosting
            # sklearn.tree.tree.DecisionTree... -> sklearn.tree._classes.DecisionTree...
            fixture_name = (
                f"{sentinel}sklearn.model_selection._search.RandomizedSearchCV("
                "estimator=sklearn.pipeline.Pipeline("
                f"ohe=sklearn.preprocessing.{module_name_encoder}.OneHotEncoder,"
                "scaler=sklearn.preprocessing._data.StandardScaler,"
                "fu=sklearn.pipeline.FeatureUnion("
                "pca=sklearn.decomposition._truncated_svd.TruncatedSVD,"
                "fs="
                "sklearn.feature_selection._univariate_selection.SelectPercentile),"
                "boosting=sklearn.ensemble._weight_boosting.AdaBoostClassifier("
                f"{estimator_name}=sklearn.tree._classes.DecisionTreeClassifier)))"
            )
        assert new_flow.name == fixture_name
        new_flow.model.fit(X, y)


# ---------------------------------------------------------------------------
# Module-level mocked tests replacing former @production_server / @test_server
# tests.  Each uses @mock.patch.object(requests.Session, ...) consistent with
# the test_delete_flow_* pattern already present in test_flow_functions.py.
# ---------------------------------------------------------------------------


def _mock_get_response(filepath):
    """Build a fake ``requests.Response`` from a fixture XML file."""
    return create_request_response(status_code=200, content_filepath=filepath)


@mock.patch.object(requests.Session, "get")
def test_get_flow(mock_get, test_files_directory):
    """Offline replacement of the former production-server test_get_flow."""
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "flows" / "flow_4024.xml"
    mock_get.return_value = _mock_get_response(content_file)

    flow = openml.flows.get_flow(4024)
    assert isinstance(flow, openml.OpenMLFlow)
    assert flow.flow_id == 4024
    assert len(flow.parameters) == 24
    assert len(flow.components) == 1

    subflow_1 = next(iter(flow.components.values()))
    assert isinstance(subflow_1, openml.OpenMLFlow)
    assert subflow_1.flow_id == 4025
    assert len(subflow_1.parameters) == 14
    assert subflow_1.parameters["E"] == "CC"
    assert len(subflow_1.components) == 1

    subflow_2 = next(iter(subflow_1.components.values()))
    assert isinstance(subflow_2, openml.OpenMLFlow)
    assert subflow_2.flow_id == 4026
    assert len(subflow_2.parameters) == 13
    assert subflow_2.parameters["I"] == "10"
    assert len(subflow_2.components) == 1

    subflow_3 = next(iter(subflow_2.components.values()))
    assert isinstance(subflow_3, openml.OpenMLFlow)
    assert subflow_3.flow_id == 1724
    assert len(subflow_3.parameters) == 11
    assert subflow_3.parameters["L"] == "-1"
    assert len(subflow_3.components) == 0


@mock.patch.object(requests.Session, "get")
def test_get_structure(mock_get, test_files_directory):
    """Offline replacement of the former production-server test_get_structure."""
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "flows" / "flow_4024.xml"
    mock_get.return_value = _mock_get_response(content_file)

    flow = openml.flows.get_flow(4024)
    flow_structure_name = flow.get_structure("name")
    flow_structure_id = flow.get_structure("flow_id")
    # components: root (filteredclassifier), multisearch, logitboost, reptree
    assert len(flow_structure_name) == 4
    assert len(flow_structure_id) == 4

    for sub_flow_name, structure in flow_structure_name.items():
        if len(structure) > 0:  # skip root element
            subflow = flow.get_subflow(structure)
            assert subflow.name == sub_flow_name

    for sub_flow_id, structure in flow_structure_id.items():
        if len(structure) > 0:  # skip root element
            subflow = flow.get_subflow(structure)
            assert subflow.flow_id == sub_flow_id


@mock.patch.object(requests.Session, "post")
@mock.patch.object(requests.Session, "get")
def test_tagging(mock_get, mock_post, test_files_directory, test_api_key):
    """Offline replacement of the former test-server test_tagging."""
    openml.config.start_using_configuration_for_example()
    fixtures = test_files_directory / "mock_responses" / "flows"

    # list_flows(size=1) -> one flow returned
    flow_list_resp = _mock_get_response(fixtures / "flow_list_1.xml")
    # get_flow(100) -> flow detail
    flow_detail_resp = _mock_get_response(fixtures / "flow_100.xml")
    # list_flows(tag=tag) with no result -> server returns 372 (NoResult)
    no_result_resp = create_request_response(
        status_code=412,
        content_filepath=fixtures / "flow_list_no_result.xml",
    )
    # list_flows(tag=tag) with one result -> flow found
    tagged_resp = _mock_get_response(fixtures / "flow_list_tagged.xml")

    # push_tag / remove_tag responses
    tag_resp = _mock_get_response(fixtures / "flow_tag.xml")
    untag_resp = _mock_get_response(fixtures / "flow_untag.xml")

    # Sequence: list_flows(size=1), get_flow(100),
    #           list_flows(tag=...) -> no result,
    #           list_flows(tag=...) -> one result,
    #           list_flows(tag=...) -> no result
    mock_get.side_effect = [flow_list_resp, flow_detail_resp, no_result_resp, tagged_resp, no_result_resp]
    mock_post.side_effect = [tag_resp, untag_resp]

    flows = openml.flows.list_flows(size=1)
    flow_id = flows["id"].iloc[0]
    flow = openml.flows.get_flow(flow_id)

    tag = "test_tag_TestFlow_1234567890"
    flows = openml.flows.list_flows(tag=tag)
    assert len(flows) == 0

    flow.push_tag(tag)
    flows = openml.flows.list_flows(tag=tag)
    assert len(flows) == 1
    assert flow_id in flows["id"].values

    flow.remove_tag(tag)
    flows = openml.flows.list_flows(tag=tag)
    assert len(flows) == 0


@mock.patch.object(requests.Session, "get")
def test_from_xml_to_xml(mock_get, test_files_directory):
    """Offline replacement of the former test-server test_from_xml_to_xml.

    Instead of fetching multiple flows from the server, we use a single
    fixture and verify the XML round-trip (parse -> serialize) is lossless.
    """
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "flows" / "flow_3.xml"
    mock_get.return_value = _mock_get_response(content_file)

    flow_xml = _perform_api_call("flow/3", request_method="get")
    flow_dict = xmltodict.parse(flow_xml)

    flow = openml.OpenMLFlow._from_dict(flow_dict)
    new_xml = flow._to_xml()

    flow_xml = (
        flow_xml.replace("  ", "")
        .replace("\t", "")
        .strip()
        .replace("\n\n", "\n")
        .replace("&quot;", '"')
    )
    flow_xml = re.sub(r"^$", "", flow_xml)
    new_xml = (
        new_xml.replace("  ", "")
        .replace("\t", "")
        .strip()
        .replace("\n\n", "\n")
        .replace("&quot;", '"')
    )
    new_xml = re.sub(r"^$", "", new_xml)

    assert new_xml == flow_xml


@pytest.mark.sklearn()
def test_publish_flow(test_files_directory, test_api_key):
    """Offline replacement of the former test-server test_publish_flow."""
    openml.config.start_using_configuration_for_example()
    extension = SklearnExtension()

    flow = openml.OpenMLFlow(
        name="sklearn.dummy.DummyClassifier",
        class_name="sklearn.dummy.DummyClassifier",
        description="test description",
        model=sklearn.dummy.DummyClassifier(),
        components=collections.OrderedDict(),
        parameters=collections.OrderedDict(),
        parameters_meta_info=collections.OrderedDict(),
        external_version=extension._format_external_version(
            "sklearn",
            sklearn.__version__,
        ),
        tags=[],
        language="English",
        dependencies=None,
    )

    with mock.patch("openml.flows.functions.flow_exists") as fe_mock, \
         mock.patch("openml.flows.functions.get_flow") as gf_mock, \
         mock.patch("openml._api_calls._perform_api_call") as api_mock:

        fe_mock.return_value = False
        api_mock.return_value = "<oml:upload_flow>\n    <oml:id>42</oml:id>\n</oml:upload_flow>"

        # After publish, get_flow is called to verify; return a copy of the flow
        published_copy = copy.deepcopy(flow)
        published_copy.flow_id = 42
        published_copy.upload_date = "2025-01-01T00:00:00"
        published_copy.version = "1"
        published_copy.uploader = "1"
        gf_mock.return_value = published_copy

        flow.publish()
        assert isinstance(flow.flow_id, int)
        assert flow.flow_id == 42


@pytest.mark.sklearn()
def test_publish_flow_with_similar_components(test_files_directory, test_api_key):
    """Offline replacement of the former test-server test_publish_flow_with_similar_components."""
    openml.config.start_using_configuration_for_example()
    extension = SklearnExtension()

    clf = sklearn.ensemble.VotingClassifier(
        [("lr", sklearn.linear_model.LogisticRegression(solver="lbfgs"))],
    )
    flow = extension.model_to_flow(clf)

    with mock.patch("openml.flows.functions.flow_exists") as fe_mock, \
         mock.patch("openml.flows.functions.get_flow") as gf_mock, \
         mock.patch("openml._api_calls._perform_api_call") as api_mock:

        api_mock.return_value = "<oml:upload_flow>\n    <oml:id>10</oml:id>\n</oml:upload_flow>"

        # First publish: flow does not exist yet
        fe_mock.return_value = False
        published_copy = copy.deepcopy(flow)
        published_copy.flow_id = 10
        published_copy.upload_date = "2025-01-01T00:00:00"
        published_copy.version = "1"
        published_copy.uploader = "1"
        for comp in published_copy.components.values():
            comp.flow_id = 11
            comp.upload_date = "2025-01-01T00:00:00"
            comp.version = "1"
            comp.uploader = "1"
        gf_mock.return_value = published_copy

        flow.publish()
        # For a flow where both components are published together, the upload
        # date should be equal
        assert flow.upload_date == flow.components["lr"].upload_date

        # Second publish with a different tree-based component
        clf2 = sklearn.ensemble.VotingClassifier(
            [("dt", sklearn.tree.DecisionTreeClassifier(max_depth=2))],
        )
        flow2 = extension.model_to_flow(clf2)
        fe_mock.return_value = False
        api_mock.return_value = "<oml:upload_flow>\n    <oml:id>20</oml:id>\n</oml:upload_flow>"
        published_copy2 = copy.deepcopy(flow2)
        published_copy2.flow_id = 20
        published_copy2.upload_date = "2025-01-01T00:01:00"
        published_copy2.version = "1"
        published_copy2.uploader = "1"
        for comp in published_copy2.components.values():
            comp.flow_id = 21
            comp.upload_date = "2025-01-01T00:00:00"
            comp.version = "1"
            comp.uploader = "1"
        gf_mock.return_value = published_copy2

        flow2.publish()
        # If one component was published before the other, the components in
        # the flow should have different upload dates
        assert flow2.upload_date != flow2.components["dt"].upload_date


@pytest.mark.sklearn()
def test_semi_legal_flow():
    """Offline replacement of the former test-server test_semi_legal_flow.

    Verifies that a nested BaggingClassifier(BaggingClassifier(DecisionTreeClassifier))
    can be converted to a flow without error. The publish step is mocked.
    """
    extension = SklearnExtension()
    estimator_name = (
        "base_estimator" if Version(sklearn.__version__) < Version("1.4") else "estimator"
    )
    semi_legal = sklearn.ensemble.BaggingClassifier(
        **{
            estimator_name: sklearn.ensemble.BaggingClassifier(
                **{
                    estimator_name: sklearn.tree.DecisionTreeClassifier(),
                }
            )
        }
    )
    flow = extension.model_to_flow(semi_legal)

    with mock.patch("openml.flows.functions.flow_exists") as fe_mock, \
         mock.patch("openml.flows.functions.get_flow") as gf_mock, \
         mock.patch("openml._api_calls._perform_api_call") as api_mock:

        fe_mock.return_value = False
        api_mock.return_value = "<oml:upload_flow>\n    <oml:id>99</oml:id>\n</oml:upload_flow>"
        published_copy = copy.deepcopy(flow)
        published_copy.flow_id = 99
        published_copy.upload_date = "2025-01-01T00:00:00"
        published_copy.version = "1"
        published_copy.uploader = "1"
        # Set IDs on all sub-components
        _set_flow_ids(published_copy, start_id=100)
        gf_mock.return_value = published_copy

        flow.publish()
        assert flow.flow_id == 99


@mock.patch.object(requests.Session, "post")
def test_nonexisting_flow_exists(mock_post, test_files_directory, test_api_key):
    """Offline replacement of the former test-server test_nonexisting_flow_exists."""
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "flows" / "flow_exists_no.xml"
    mock_post.return_value = _mock_get_response(content_file)

    flow_id = openml.flows.flow_exists("TESTnonexistent_flow_name", "TESTnonexistent_version")
    assert not flow_id


@mock.patch.object(requests.Session, "post")
def test_existing_flow_exists(mock_post, test_files_directory, test_api_key):
    """Offline replacement of the former test-server test_existing_flow_exists."""
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "flows" / "flow_exists_yes.xml"
    mock_post.return_value = _mock_get_response(content_file)

    flow_id = openml.flows.flow_exists("some.existing.flow", "1.0")
    assert flow_id == 42


@mock.patch.object(requests.Session, "get")
def test_download_non_scikit_learn_flows(mock_get, test_files_directory):
    """Offline replacement of the former production-server test_download_non_scikit_learn_flows."""
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "flows" / "flow_6742.xml"
    mock_get.return_value = _mock_get_response(content_file)

    flow = openml.flows.get_flow(6742)
    assert isinstance(flow, openml.OpenMLFlow)
    assert flow.flow_id == 6742
    assert len(flow.parameters) == 19
    assert len(flow.components) == 1
    assert flow.model is None

    subflow_1 = next(iter(flow.components.values()))
    assert isinstance(subflow_1, openml.OpenMLFlow)
    assert subflow_1.flow_id == 6743
    assert len(subflow_1.parameters) == 8
    assert subflow_1.parameters["U"] == "0"
    assert len(subflow_1.components) == 1
    assert subflow_1.model is None

    subflow_2 = next(iter(subflow_1.components.values()))
    assert isinstance(subflow_2, openml.OpenMLFlow)
    assert subflow_2.flow_id == 5888
    assert len(subflow_2.parameters) == 4
    assert subflow_2.parameters["batch-size"] is None
    assert len(subflow_2.components) == 0
    assert subflow_2.model is None


# ---------------------------------------------------------------------------
# Helpers for mocked tests
# ---------------------------------------------------------------------------

def _set_flow_ids(flow, start_id=100):
    """Recursively set flow_id, upload_date, version, uploader on sub-components."""
    counter = start_id
    for comp in flow.components.values():
        comp.flow_id = counter
        comp.upload_date = "2025-01-01T00:00:00"
        comp.version = "1"
        comp.uploader = "1"
        counter += 1
        counter = _set_flow_ids(comp, counter)
    return counter
