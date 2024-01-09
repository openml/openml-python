# License: BSD 3-Clause
from __future__ import annotations

import collections
import copy
import hashlib
import re
import time
from distutils.version import LooseVersion
from unittest import mock

import pytest
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

import openml
import openml.exceptions
import openml.extensions.sklearn
import openml.utils
from openml._api_calls import _perform_api_call
from openml.testing import SimpleImputer, TestBase


class TestFlow(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()
        self.extension = openml.extensions.sklearn.SklearnExtension()

    def tearDown(self):
        super().tearDown()

    @pytest.mark.production()
    def test_get_flow(self):
        # We need to use the production server here because 4024 is not the
        # test server
        openml.config.server = self.production_server

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

    @pytest.mark.production()
    def test_get_structure(self):
        # also responsible for testing: flow.get_subflow
        # We need to use the production server here because 4024 is not the
        # test server
        openml.config.server = self.production_server

        flow = openml.flows.get_flow(4024)
        flow_structure_name = flow.get_structure("name")
        flow_structure_id = flow.get_structure("flow_id")
        # components: root (filteredclassifier), multisearch, loginboost,
        # reptree
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

    def test_tagging(self):
        flows = openml.flows.list_flows(size=1, output_format="dataframe")
        flow_id = flows["id"].iloc[0]
        flow = openml.flows.get_flow(flow_id)
        # tags can be at most 64 alphanumeric (+ underscore) chars
        unique_indicator = str(time.time()).replace(".", "")
        tag = f"test_tag_TestFlow_{unique_indicator}"
        flows = openml.flows.list_flows(tag=tag, output_format="dataframe")
        assert len(flows) == 0
        flow.push_tag(tag)
        flows = openml.flows.list_flows(tag=tag, output_format="dataframe")
        assert len(flows) == 1
        assert flow_id in flows["id"]
        flow.remove_tag(tag)
        flows = openml.flows.list_flows(tag=tag, output_format="dataframe")
        assert len(flows) == 0

    def test_from_xml_to_xml(self):
        # Get the raw xml thing
        # TODO maybe get this via get_flow(), which would have to be refactored
        # to allow getting only the xml dictionary
        # TODO: no sklearn flows.
        for flow_id in [
            3,
            5,
            7,
            9,
        ]:
            flow_xml = _perform_api_call("flow/%d" % flow_id, request_method="get")
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
    def test_to_xml_from_xml(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier(),
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
    def test_publish_flow(self):
        flow = openml.OpenMLFlow(
            name="sklearn.dummy.DummyClassifier",
            class_name="sklearn.dummy.DummyClassifier",
            description="test description",
            model=sklearn.dummy.DummyClassifier(),
            components=collections.OrderedDict(),
            parameters=collections.OrderedDict(),
            parameters_meta_info=collections.OrderedDict(),
            external_version=self.extension._format_external_version(
                "sklearn",
                sklearn.__version__,
            ),
            tags=[],
            language="English",
            dependencies=None,
        )

        flow, _ = self._add_sentinel_to_flow_name(flow, None)

        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))
        assert isinstance(flow.flow_id, int)

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
            "collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id),
        )

    @pytest.mark.sklearn()
    def test_publish_flow_with_similar_components(self):
        clf = sklearn.ensemble.VotingClassifier(
            [("lr", sklearn.linear_model.LogisticRegression(solver="lbfgs"))],
        )
        flow = self.extension.model_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))
        # For a flow where both components are published together, the upload
        # date should be equal
        assert flow.upload_date == flow.components["lr"].upload_date, (
            flow.name,
            flow.flow_id,
            flow.components["lr"].name,
            flow.components["lr"].flow_id,
        )

        clf1 = sklearn.tree.DecisionTreeClassifier(max_depth=2)
        flow1 = self.extension.model_to_flow(clf1)
        flow1, sentinel = self._add_sentinel_to_flow_name(flow1, None)
        flow1.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow1.flow_id))

        # In order to assign different upload times to the flows!
        time.sleep(1)

        clf2 = sklearn.ensemble.VotingClassifier(
            [("dt", sklearn.tree.DecisionTreeClassifier(max_depth=2))],
        )
        flow2 = self.extension.model_to_flow(clf2)
        flow2, _ = self._add_sentinel_to_flow_name(flow2, sentinel)
        flow2.publish()
        TestBase._mark_entity_for_removal("flow", flow2.flow_id, flow2.name)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow2.flow_id))
        # If one component was published before the other, the components in
        # the flow should have different upload dates
        assert flow2.upload_date != flow2.components["dt"].upload_date

        clf3 = sklearn.ensemble.AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=3))
        flow3 = self.extension.model_to_flow(clf3)
        flow3, _ = self._add_sentinel_to_flow_name(flow3, sentinel)
        # Child flow has different parameter. Check for storing the flow
        # correctly on the server should thus not check the child's parameters!
        flow3.publish()
        TestBase._mark_entity_for_removal("flow", flow3.flow_id, flow3.name)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow3.flow_id))

    @pytest.mark.sklearn()
    def test_semi_legal_flow(self):
        # TODO: Test if parameters are set correctly!
        # should not throw error as it contains two differentiable forms of
        # Bagging i.e., Bagging(Bagging(J48)) and Bagging(J48)
        semi_legal = sklearn.ensemble.BaggingClassifier(
            base_estimator=sklearn.ensemble.BaggingClassifier(
                base_estimator=sklearn.tree.DecisionTreeClassifier(),
            ),
        )
        flow = self.extension.model_to_flow(semi_legal)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)

        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))

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

        if LooseVersion(sklearn.__version__) < "0.22":
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
            "collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id),
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

    def test_nonexisting_flow_exists(self):
        def get_sentinel():
            # Create a unique prefix for the flow. Necessary because the flow
            # is identified by its name and external version online. Having a
            # unique name allows us to publish the same flow in each test run
            md5 = hashlib.md5()
            md5.update(str(time.time()).encode("utf-8"))
            sentinel = md5.hexdigest()[:10]
            return "TEST%s" % sentinel

        name = get_sentinel() + get_sentinel()
        version = get_sentinel()

        flow_id = openml.flows.flow_exists(name, version)
        assert not flow_id

    @pytest.mark.sklearn()
    def test_existing_flow_exists(self):
        # create a flow
        nb = sklearn.naive_bayes.GaussianNB()

        ohe_params = {"sparse": False, "handle_unknown": "ignore"}
        if LooseVersion(sklearn.__version__) >= "0.20":
            ohe_params["categories"] = "auto"
        steps = [
            ("imputation", SimpleImputer(strategy="median")),
            ("hotencoding", sklearn.preprocessing.OneHotEncoder(**ohe_params)),
            (
                "variencethreshold",
                sklearn.feature_selection.VarianceThreshold(),
            ),
            ("classifier", sklearn.tree.DecisionTreeClassifier()),
        ]
        complicated = sklearn.pipeline.Pipeline(steps=steps)

        for classifier in [nb, complicated]:
            flow = self.extension.model_to_flow(classifier)
            flow, _ = self._add_sentinel_to_flow_name(flow, None)
            # publish the flow
            flow = flow.publish()
            TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
            TestBase.logger.info(
                "collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id),
            )
            # redownload the flow
            flow = openml.flows.get_flow(flow.flow_id)

            # check if flow exists can find it
            flow = openml.flows.get_flow(flow.flow_id)
            downloaded_flow_id = openml.flows.flow_exists(
                flow.name,
                flow.external_version,
            )
            assert downloaded_flow_id == flow.flow_id

    @pytest.mark.sklearn()
    def test_sklearn_to_upload_to_flow(self):
        iris = sklearn.datasets.load_iris()
        X = iris.data
        y = iris.target

        # Test a more complicated flow
        ohe_params = {"handle_unknown": "ignore"}
        if LooseVersion(sklearn.__version__) >= "0.20":
            ohe_params["categories"] = "auto"
        ohe = sklearn.preprocessing.OneHotEncoder(**ohe_params)
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        pca = sklearn.decomposition.TruncatedSVD()
        fs = sklearn.feature_selection.SelectPercentile(
            score_func=sklearn.feature_selection.f_classif,
            percentile=30,
        )
        fu = sklearn.pipeline.FeatureUnion(transformer_list=[("pca", pca), ("fs", fs)])
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier(),
        )
        model = sklearn.pipeline.Pipeline(
            steps=[("ohe", ohe), ("scaler", scaler), ("fu", fu), ("boosting", boosting)],
        )
        parameter_grid = {
            "boosting__n_estimators": [1, 5, 10, 100],
            "boosting__learning_rate": scipy.stats.uniform(0.01, 0.99),
            "boosting__base_estimator__max_depth": scipy.stats.randint(1, 10),
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
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))
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
        module_name_encoder = "_encoders" if LooseVersion(sklearn.__version__) >= "0.20" else "data"
        if LooseVersion(sklearn.__version__) < "0.22":
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
                "base_estimator=sklearn.tree._classes.DecisionTreeClassifier)))"
            )
        assert new_flow.name == fixture_name
        new_flow.model.fit(X, y)

    def test_extract_tags(self):
        flow_xml = "<oml:tag>study_14</oml:tag>"
        flow_dict = xmltodict.parse(flow_xml)
        tags = openml.utils.extract_xml_tags("oml:tag", flow_dict)
        assert tags == ["study_14"]

        flow_xml = "<oml:flow><oml:tag>OpenmlWeka</oml:tag>\n" "<oml:tag>weka</oml:tag></oml:flow>"
        flow_dict = xmltodict.parse(flow_xml)
        tags = openml.utils.extract_xml_tags("oml:tag", flow_dict["oml:flow"])
        assert tags == ["OpenmlWeka", "weka"]

    @pytest.mark.production()
    def test_download_non_scikit_learn_flows(self):
        openml.config.server = self.production_server

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
