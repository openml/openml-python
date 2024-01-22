# License: BSD 3-Clause
from __future__ import annotations

import collections
import json
import os
import re
import sys
import unittest
import warnings
from collections import OrderedDict
from distutils.version import LooseVersion
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import scipy.stats
import sklearn.base
import sklearn.cluster
import sklearn.datasets
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree
from packaging import version
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import openml
from openml.exceptions import PyOpenMLError
from openml.extensions.sklearn import SklearnExtension, cat, cont
from openml.flows import OpenMLFlow
from openml.flows.functions import assert_flows_equal
from openml.runs.trace import OpenMLRunTrace
from openml.testing import CustomImputer, SimpleImputer, TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


__version__ = 0.1


class Model(sklearn.base.BaseEstimator):
    def __init__(self, boolean, integer, floating_point_value):
        self.boolean = boolean
        self.integer = integer
        self.floating_point_value = floating_point_value

    def fit(self, X, y):
        pass


class TestSklearnExtensionFlowFunctions(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take less
    # than 1 seconds

    def setUp(self):
        super().setUp(n_levels=2)
        iris = sklearn.datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

        self.extension = SklearnExtension()

    def _get_expected_pipeline_description(self, model: Any) -> str:
        if version.parse(sklearn.__version__) >= version.parse("1.0"):
            expected_fixture = (
                "Pipeline of transforms with a final estimator.\n\nSequentially"
                " apply a list of transforms and a final estimator.\n"
                "Intermediate steps of the pipeline must be 'transforms', that "
                "is, they\nmust implement `fit` and `transform` methods.\nThe final "
                "estimator only needs to implement `fit`.\nThe transformers in "
                "the pipeline can be cached using ``memory`` argument.\n\nThe "
                "purpose of the pipeline is to assemble several steps that can "
                "be\ncross-validated together while setting different parameters"
                ". For this, it\nenables setting parameters of the various steps"
                " using their names and the\nparameter name separated by a `'__'`,"
                " as in the example below. A step's\nestimator may be replaced "
                "entirely by setting the parameter with its name\nto another "
                "estimator, or a transformer removed by setting it to\n"
                "`'passthrough'` or `None`."
            )
        elif version.parse(sklearn.__version__) >= version.parse("0.21.0"):
            expected_fixture = (
                "Pipeline of transforms with a final estimator.\n\nSequentially"
                " apply a list of transforms and a final estimator.\n"
                "Intermediate steps of the pipeline must be 'transforms', that "
                "is, they\nmust implement fit and transform methods.\nThe final "
                "estimator only needs to implement fit.\nThe transformers in "
                "the pipeline can be cached using ``memory`` argument.\n\nThe "
                "purpose of the pipeline is to assemble several steps that can "
                "be\ncross-validated together while setting different parameters"
                ".\nFor this, it enables setting parameters of the various steps"
                " using their\nnames and the parameter name separated by a '__',"
                " as in the example below.\nA step's estimator may be replaced "
                "entirely by setting the parameter\nwith its name to another "
                "estimator, or a transformer removed by setting\nit to "
                "'passthrough' or ``None``."
            )
        else:
            expected_fixture = self.extension._get_sklearn_description(model)
        return expected_fixture

    def _serialization_test_helper(
        self,
        model,
        X,
        y,
        subcomponent_parameters,
        dependencies_mock_call_count=(1, 2),
    ):
        # Regex pattern for memory addresses of style 0x7f8e0f31ecf8
        pattern = re.compile("0x[0-9a-f]{12}")

        with mock.patch.object(self.extension, "_check_dependencies") as check_dependencies_mock:
            serialization = self.extension.model_to_flow(model)

            if X is not None:
                model.fit(X, y)

            new_model = self.extension.flow_to_model(serialization)
            # compares string representations of the dict, as it potentially
            # contains complex objects that can not be compared with == op
            assert re.sub(pattern, str(model.get_params()), "") == re.sub(
                pattern, str(new_model.get_params()), ""
            )

            assert type(new_model) == type(model)
            assert new_model is not model

            if X is not None:
                new_model.fit(self.X, self.y)

            assert check_dependencies_mock.call_count == dependencies_mock_call_count[0]

            xml = serialization._to_dict()
            new_model2 = self.extension.flow_to_model(OpenMLFlow._from_dict(xml))
            assert re.sub(pattern, str(model.get_params()), "") == re.sub(
                pattern, str(new_model2.get_params()), ""
            )

            assert type(new_model2) == type(model)
            assert new_model2 is not model

            if X is not None:
                new_model2.fit(self.X, self.y)

            assert check_dependencies_mock.call_count == dependencies_mock_call_count[1]

            if subcomponent_parameters:
                for nm in (new_model, new_model2):
                    new_model_params = nm.get_params()
                    model_params = model.get_params()
                    for subcomponent_parameter in subcomponent_parameters:
                        assert type(new_model_params[subcomponent_parameter]) == type(
                            model_params[subcomponent_parameter]
                        )
                        assert (
                            new_model_params[subcomponent_parameter]
                            is not model_params[subcomponent_parameter]
                        )
                        del new_model_params[subcomponent_parameter]
                        del model_params[subcomponent_parameter]
                    assert new_model_params == model_params

            return serialization, new_model

    @pytest.mark.sklearn()
    def test_serialize_model(self):
        model = sklearn.tree.DecisionTreeClassifier(
            criterion="entropy",
            max_features="auto",
            max_leaf_nodes=2000,
        )

        tree_name = "tree" if LooseVersion(sklearn.__version__) < "0.22" else "_classes"
        fixture_name = f"sklearn.tree.{tree_name}.DecisionTreeClassifier"
        fixture_short_name = "sklearn.DecisionTreeClassifier"
        # str obtained from self.extension._get_sklearn_description(model)
        fixture_description = "A decision tree classifier."
        version_fixture = self.extension._min_dependency_str(sklearn.__version__)

        presort_val = "false" if LooseVersion(sklearn.__version__) < "0.22" else '"deprecated"'
        # min_impurity_decrease has been introduced in 0.20
        # min_impurity_split has been deprecated in 0.20
        if LooseVersion(sklearn.__version__) < "0.19":
            fixture_parameters = OrderedDict(
                (
                    ("class_weight", "null"),
                    ("criterion", '"entropy"'),
                    ("max_depth", "null"),
                    ("max_features", '"auto"'),
                    ("max_leaf_nodes", "2000"),
                    ("min_impurity_split", "1e-07"),
                    ("min_samples_leaf", "1"),
                    ("min_samples_split", "2"),
                    ("min_weight_fraction_leaf", "0.0"),
                    ("presort", "false"),
                    ("random_state", "null"),
                    ("splitter", '"best"'),
                ),
            )
        elif LooseVersion(sklearn.__version__) < "1.0":
            fixture_parameters = OrderedDict(
                (
                    ("class_weight", "null"),
                    ("criterion", '"entropy"'),
                    ("max_depth", "null"),
                    ("max_features", '"auto"'),
                    ("max_leaf_nodes", "2000"),
                    ("min_impurity_decrease", "0.0"),
                    ("min_impurity_split", "null"),
                    ("min_samples_leaf", "1"),
                    ("min_samples_split", "2"),
                    ("min_weight_fraction_leaf", "0.0"),
                    ("presort", presort_val),
                    ("random_state", "null"),
                    ("splitter", '"best"'),
                ),
            )
        else:
            fixture_parameters = OrderedDict(
                (
                    ("class_weight", "null"),
                    ("criterion", '"entropy"'),
                    ("max_depth", "null"),
                    ("max_features", '"auto"'),
                    ("max_leaf_nodes", "2000"),
                    ("min_impurity_decrease", "0.0"),
                    ("min_samples_leaf", "1"),
                    ("min_samples_split", "2"),
                    ("min_weight_fraction_leaf", "0.0"),
                    ("presort", presort_val),
                    ("random_state", "null"),
                    ("splitter", '"best"'),
                ),
            )

        if LooseVersion(sklearn.__version__) >= "0.22":
            fixture_parameters.update({"ccp_alpha": "0.0"})
            fixture_parameters.move_to_end("ccp_alpha", last=False)
        if LooseVersion(sklearn.__version__) >= "0.24":
            del fixture_parameters["presort"]

        structure_fixture = {f"sklearn.tree.{tree_name}.DecisionTreeClassifier": []}

        serialization, _ = self._serialization_test_helper(
            model,
            X=self.X,
            y=self.y,
            subcomponent_parameters=None,
        )
        structure = serialization.get_structure("name")

        assert serialization.name == fixture_name
        assert serialization.class_name == fixture_name
        assert serialization.custom_name == fixture_short_name
        assert serialization.description == fixture_description
        assert serialization.parameters == fixture_parameters
        assert serialization.dependencies == version_fixture
        self.assertDictEqual(structure, structure_fixture)

    @pytest.mark.sklearn()
    @pytest.mark.production()
    def test_can_handle_flow(self):
        openml.config.server = self.production_server

        R_flow = openml.flows.get_flow(6794)
        assert not self.extension.can_handle_flow(R_flow)
        old_3rd_party_flow = openml.flows.get_flow(7660)
        assert self.extension.can_handle_flow(old_3rd_party_flow)

        openml.config.server = self.test_server

    @pytest.mark.sklearn()
    def test_serialize_model_clustering(self):
        model = sklearn.cluster.KMeans()

        cluster_name = "k_means_" if LooseVersion(sklearn.__version__) < "0.22" else "_kmeans"
        fixture_name = f"sklearn.cluster.{cluster_name}.KMeans"
        fixture_short_name = "sklearn.KMeans"
        # str obtained from self.extension._get_sklearn_description(model)
        fixture_description = "K-Means clustering{}".format(
            "" if LooseVersion(sklearn.__version__) < "0.22" else ".",
        )
        version_fixture = self.extension._min_dependency_str(sklearn.__version__)

        n_jobs_val = "null" if LooseVersion(sklearn.__version__) < "0.23" else '"deprecated"'
        precomp_val = '"auto"' if LooseVersion(sklearn.__version__) < "0.23" else '"deprecated"'

        # n_jobs default has changed to None in 0.20
        if LooseVersion(sklearn.__version__) < "0.20":
            fixture_parameters = OrderedDict(
                (
                    ("algorithm", '"auto"'),
                    ("copy_x", "true"),
                    ("init", '"k-means++"'),
                    ("max_iter", "300"),
                    ("n_clusters", "8"),
                    ("n_init", "10"),
                    ("n_jobs", "1"),
                    ("precompute_distances", '"auto"'),
                    ("random_state", "null"),
                    ("tol", "0.0001"),
                    ("verbose", "0"),
                ),
            )
        elif LooseVersion(sklearn.__version__) < "1.0":
            fixture_parameters = OrderedDict(
                (
                    ("algorithm", '"auto"'),
                    ("copy_x", "true"),
                    ("init", '"k-means++"'),
                    ("max_iter", "300"),
                    ("n_clusters", "8"),
                    ("n_init", "10"),
                    ("n_jobs", n_jobs_val),
                    ("precompute_distances", precomp_val),
                    ("random_state", "null"),
                    ("tol", "0.0001"),
                    ("verbose", "0"),
                ),
            )
        elif LooseVersion(sklearn.__version__) < "1.1":
            fixture_parameters = OrderedDict(
                (
                    ("algorithm", '"auto"'),
                    ("copy_x", "true"),
                    ("init", '"k-means++"'),
                    ("max_iter", "300"),
                    ("n_clusters", "8"),
                    ("n_init", "10"),
                    ("random_state", "null"),
                    ("tol", "0.0001"),
                    ("verbose", "0"),
                ),
            )
        else:
            n_init = '"warn"' if LooseVersion(sklearn.__version__) >= "1.2" else "10"
            fixture_parameters = OrderedDict(
                (
                    ("algorithm", '"lloyd"'),
                    ("copy_x", "true"),
                    ("init", '"k-means++"'),
                    ("max_iter", "300"),
                    ("n_clusters", "8"),
                    ("n_init", n_init),
                    ("random_state", "null"),
                    ("tol", "0.0001"),
                    ("verbose", "0"),
                ),
            )
        fixture_structure = {f"sklearn.cluster.{cluster_name}.KMeans": []}

        serialization, _ = self._serialization_test_helper(
            model,
            X=None,
            y=None,
            subcomponent_parameters=None,
        )
        structure = serialization.get_structure("name")

        assert serialization.name == fixture_name
        assert serialization.class_name == fixture_name
        assert serialization.custom_name == fixture_short_name
        assert serialization.description == fixture_description
        assert serialization.parameters == fixture_parameters
        assert serialization.dependencies == version_fixture
        assert structure == fixture_structure

    @pytest.mark.sklearn()
    def test_serialize_model_with_subcomponent(self):
        model = sklearn.ensemble.AdaBoostClassifier(
            n_estimators=100,
            base_estimator=sklearn.tree.DecisionTreeClassifier(),
        )

        weight_name = "{}weight_boosting".format(
            "" if LooseVersion(sklearn.__version__) < "0.22" else "_",
        )
        tree_name = "tree" if LooseVersion(sklearn.__version__) < "0.22" else "_classes"
        fixture_name = (
            f"sklearn.ensemble.{weight_name}.AdaBoostClassifier"
            f"(base_estimator=sklearn.tree.{tree_name}.DecisionTreeClassifier)"
        )
        fixture_class_name = f"sklearn.ensemble.{weight_name}.AdaBoostClassifier"
        fixture_short_name = "sklearn.AdaBoostClassifier"
        # str obtained from self.extension._get_sklearn_description(model)
        fixture_description = (
            "An AdaBoost classifier.\n\nAn AdaBoost [1] classifier is a "
            "meta-estimator that begins by fitting a\nclassifier on the original"
            " dataset and then fits additional copies of the\nclassifier on the "
            "same dataset but where the weights of incorrectly\nclassified "
            "instances are adjusted such that subsequent classifiers focus\nmore"
            " on difficult cases.\n\nThis class implements the algorithm known "
            "as AdaBoost-SAMME [2]."
        )
        fixture_subcomponent_name = f"sklearn.tree.{tree_name}.DecisionTreeClassifier"
        fixture_subcomponent_class_name = f"sklearn.tree.{tree_name}.DecisionTreeClassifier"
        # str obtained from self.extension._get_sklearn_description(model.base_estimator)
        fixture_subcomponent_description = "A decision tree classifier."
        fixture_structure = {
            fixture_name: [],
            f"sklearn.tree.{tree_name}.DecisionTreeClassifier": ["base_estimator"],
        }

        serialization, _ = self._serialization_test_helper(
            model,
            X=self.X,
            y=self.y,
            subcomponent_parameters=["base_estimator"],
            dependencies_mock_call_count=(2, 4),
        )
        structure = serialization.get_structure("name")

        assert serialization.name == fixture_name
        assert serialization.class_name == fixture_class_name
        assert serialization.custom_name == fixture_short_name
        assert serialization.description == fixture_description
        assert serialization.parameters["algorithm"] == '"SAMME.R"'
        assert isinstance(serialization.parameters["base_estimator"], str)
        assert serialization.parameters["learning_rate"] == "1.0"
        assert serialization.parameters["n_estimators"] == "100"
        assert serialization.components["base_estimator"].name == fixture_subcomponent_name
        assert (
            serialization.components["base_estimator"].class_name == fixture_subcomponent_class_name
        )
        assert (
            serialization.components["base_estimator"].description
            == fixture_subcomponent_description
        )
        self.assertDictEqual(structure, fixture_structure)

    @pytest.mark.sklearn()
    def test_serialize_pipeline(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        dummy = sklearn.dummy.DummyClassifier(strategy="prior")
        model = sklearn.pipeline.Pipeline(steps=[("scaler", scaler), ("dummy", dummy)])

        scaler_name = "data" if LooseVersion(sklearn.__version__) < "0.22" else "_data"
        fixture_name = (
            "sklearn.pipeline.Pipeline("
            f"scaler=sklearn.preprocessing.{scaler_name}.StandardScaler,"
            "dummy=sklearn.dummy.DummyClassifier)"
        )
        fixture_short_name = "sklearn.Pipeline(StandardScaler,DummyClassifier)"
        fixture_description = self._get_expected_pipeline_description(model)
        fixture_structure = {
            fixture_name: [],
            f"sklearn.preprocessing.{scaler_name}.StandardScaler": ["scaler"],
            "sklearn.dummy.DummyClassifier": ["dummy"],
        }

        serialization, new_model = self._serialization_test_helper(
            model,
            X=self.X,
            y=self.y,
            subcomponent_parameters=["scaler", "dummy", "steps"],
            dependencies_mock_call_count=(3, 6),
        )
        structure = serialization.get_structure("name")

        assert serialization.name == fixture_name
        assert serialization.custom_name == fixture_short_name
        assert serialization.description == fixture_description
        self.assertDictEqual(structure, fixture_structure)

        # Comparing the pipeline
        # The parameters only have the name of base objects(not the whole flow)
        # as value
        # memory parameter has been added in 0.19, verbose in 0.21
        if LooseVersion(sklearn.__version__) < "0.19":
            assert len(serialization.parameters) == 1
        elif LooseVersion(sklearn.__version__) < "0.21":
            assert len(serialization.parameters) == 2
        else:
            assert len(serialization.parameters) == 3

        # Hard to compare two representations of a dict due to possibly
        # different sorting. Making a json makes it easier
        assert json.loads(serialization.parameters["steps"]) == [
            {
                "oml-python:serialized_object": "component_reference",
                "value": {"key": "scaler", "step_name": "scaler"},
            },
            {
                "oml-python:serialized_object": "component_reference",
                "value": {"key": "dummy", "step_name": "dummy"},
            },
        ]

        # Checking the sub-component
        assert len(serialization.components) == 2
        assert isinstance(serialization.components["scaler"], OpenMLFlow)
        assert isinstance(serialization.components["dummy"], OpenMLFlow)

        assert [step[0] for step in new_model.steps] == [step[0] for step in model.steps]
        assert new_model.steps[0][1] is not model.steps[0][1]
        assert new_model.steps[1][1] is not model.steps[1][1]

    @pytest.mark.sklearn()
    def test_serialize_pipeline_clustering(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        km = sklearn.cluster.KMeans()
        model = sklearn.pipeline.Pipeline(steps=[("scaler", scaler), ("clusterer", km)])

        scaler_name = "data" if LooseVersion(sklearn.__version__) < "0.22" else "_data"
        cluster_name = "k_means_" if LooseVersion(sklearn.__version__) < "0.22" else "_kmeans"
        fixture_name = (
            "sklearn.pipeline.Pipeline("
            f"scaler=sklearn.preprocessing.{scaler_name}.StandardScaler,"
            f"clusterer=sklearn.cluster.{cluster_name}.KMeans)"
        )
        fixture_short_name = "sklearn.Pipeline(StandardScaler,KMeans)"
        fixture_description = self._get_expected_pipeline_description(model)
        fixture_structure = {
            fixture_name: [],
            f"sklearn.preprocessing.{scaler_name}.StandardScaler": ["scaler"],
            f"sklearn.cluster.{cluster_name}.KMeans": ["clusterer"],
        }
        serialization, new_model = self._serialization_test_helper(
            model,
            X=None,
            y=None,
            subcomponent_parameters=["scaler", "steps", "clusterer"],
            dependencies_mock_call_count=(3, 6),
        )
        structure = serialization.get_structure("name")

        assert serialization.name == fixture_name
        assert serialization.custom_name == fixture_short_name
        assert serialization.description == fixture_description
        self.assertDictEqual(structure, fixture_structure)

        # Comparing the pipeline
        # The parameters only have the name of base objects(not the whole flow)
        # as value
        # memory parameter has been added in 0.19
        if LooseVersion(sklearn.__version__) < "0.19":
            assert len(serialization.parameters) == 1
        elif LooseVersion(sklearn.__version__) < "0.21":
            assert len(serialization.parameters) == 2
        else:
            assert len(serialization.parameters) == 3
        # Hard to compare two representations of a dict due to possibly
        # different sorting. Making a json makes it easier
        assert json.loads(serialization.parameters["steps"]) == [
            {
                "oml-python:serialized_object": "component_reference",
                "value": {"key": "scaler", "step_name": "scaler"},
            },
            {
                "oml-python:serialized_object": "component_reference",
                "value": {"key": "clusterer", "step_name": "clusterer"},
            },
        ]

        # Checking the sub-component
        assert len(serialization.components) == 2
        assert isinstance(serialization.components["scaler"], OpenMLFlow)
        assert isinstance(serialization.components["clusterer"], OpenMLFlow)

        assert [step[0] for step in new_model.steps] == [step[0] for step in model.steps]
        assert new_model.steps[0][1] is not model.steps[0][1]
        assert new_model.steps[1][1] is not model.steps[1][1]

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    def test_serialize_column_transformer(self):
        # temporary local import, dependend on version 0.20
        import sklearn.compose

        model = sklearn.compose.ColumnTransformer(
            transformers=[
                ("numeric", sklearn.preprocessing.StandardScaler(), [0, 1, 2]),
                (
                    "nominal",
                    sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
                    [3, 4, 5],
                ),
                ("drop", "drop", [6, 7, 8]),
            ],
            remainder="passthrough",
        )

        scaler_name = "data" if LooseVersion(sklearn.__version__) < "0.22" else "_data"
        fixture = (
            "sklearn.compose._column_transformer.ColumnTransformer("
            f"numeric=sklearn.preprocessing.{scaler_name}.StandardScaler,"
            "nominal=sklearn.preprocessing._encoders.OneHotEncoder,drop=drop)"
        )
        fixture_short_name = "sklearn.ColumnTransformer"

        if version.parse(sklearn.__version__) >= version.parse("0.21.0"):
            # str obtained from self.extension._get_sklearn_description(model)
            fixture_description = (
                "Applies transformers to columns of an array or pandas "
                "DataFrame.\n\nThis estimator allows different columns or "
                "column subsets of the input\nto be transformed separately and "
                "the features generated by each transformer\nwill be "
                "concatenated to form a single feature space.\nThis is useful "
                "for heterogeneous or columnar data, to combine several\nfeature"
                " extraction mechanisms or transformations into a single "
                "transformer."
            )
        else:
            fixture_description = self.extension._get_sklearn_description(model)

        fixture_structure = {
            fixture: [],
            f"sklearn.preprocessing.{scaler_name}.StandardScaler": ["numeric"],
            "sklearn.preprocessing._encoders.OneHotEncoder": ["nominal"],
            "drop": ["drop"],
        }

        serialization = self.extension.model_to_flow(model)
        structure = serialization.get_structure("name")
        assert serialization.name == fixture
        assert serialization.custom_name == fixture_short_name
        assert serialization.description == fixture_description
        self.assertDictEqual(structure, fixture_structure)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    def test_serialize_column_transformer_pipeline(self):
        # temporary local import, dependend on version 0.20
        import sklearn.compose

        inner = sklearn.compose.ColumnTransformer(
            transformers=[
                ("numeric", sklearn.preprocessing.StandardScaler(), [0, 1, 2]),
                (
                    "nominal",
                    sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
                    [3, 4, 5],
                ),
            ],
            remainder="passthrough",
        )
        model = sklearn.pipeline.Pipeline(
            steps=[("transformer", inner), ("classifier", sklearn.tree.DecisionTreeClassifier())],
        )
        scaler_name = "data" if LooseVersion(sklearn.__version__) < "0.22" else "_data"
        tree_name = "tree" if LooseVersion(sklearn.__version__) < "0.22" else "_classes"
        fixture_name = (
            "sklearn.pipeline.Pipeline("
            "transformer=sklearn.compose._column_transformer."
            "ColumnTransformer("
            f"numeric=sklearn.preprocessing.{scaler_name}.StandardScaler,"
            "nominal=sklearn.preprocessing._encoders.OneHotEncoder),"
            f"classifier=sklearn.tree.{tree_name}.DecisionTreeClassifier)"
        )
        fixture_structure = {
            f"sklearn.preprocessing.{scaler_name}.StandardScaler": [
                "transformer",
                "numeric",
            ],
            "sklearn.preprocessing._encoders.OneHotEncoder": ["transformer", "nominal"],
            "sklearn.compose._column_transformer.ColumnTransformer(numeric="
            f"sklearn.preprocessing.{scaler_name}.StandardScaler,nominal=sklearn."
            "preprocessing._encoders.OneHotEncoder)": ["transformer"],
            f"sklearn.tree.{tree_name}.DecisionTreeClassifier": ["classifier"],
            fixture_name: [],
        }

        fixture_description = self._get_expected_pipeline_description(model)
        serialization, new_model = self._serialization_test_helper(
            model,
            X=None,
            y=None,
            subcomponent_parameters=(
                "transformer",
                "classifier",
                "transformer__transformers",
                "steps",
                "transformer__nominal",
                "transformer__numeric",
            ),
            dependencies_mock_call_count=(5, 10),
        )
        structure = serialization.get_structure("name")
        assert serialization.name == fixture_name
        assert serialization.description == fixture_description

        self.assertDictEqual(structure, fixture_structure)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="Pipeline processing behaviour updated",
    )
    def test_serialize_feature_union(self):
        ohe_params = {"sparse": False}
        if LooseVersion(sklearn.__version__) >= "0.20":
            ohe_params["categories"] = "auto"
        ohe = sklearn.preprocessing.OneHotEncoder(**ohe_params)
        scaler = sklearn.preprocessing.StandardScaler()

        fu = sklearn.pipeline.FeatureUnion(transformer_list=[("ohe", ohe), ("scaler", scaler)])
        serialization, new_model = self._serialization_test_helper(
            fu,
            X=self.X,
            y=self.y,
            subcomponent_parameters=("ohe", "scaler", "transformer_list"),
            dependencies_mock_call_count=(3, 6),
        )
        structure = serialization.get_structure("name")
        # OneHotEncoder was moved to _encoders module in 0.20
        module_name_encoder = "_encoders" if LooseVersion(sklearn.__version__) >= "0.20" else "data"
        scaler_name = "data" if LooseVersion(sklearn.__version__) < "0.22" else "_data"
        fixture_name = (
            "sklearn.pipeline.FeatureUnion("
            f"ohe=sklearn.preprocessing.{module_name_encoder}.OneHotEncoder,"
            f"scaler=sklearn.preprocessing.{scaler_name}.StandardScaler)"
        )
        fixture_structure = {
            fixture_name: [],
            f"sklearn.preprocessing.{module_name_encoder}." "OneHotEncoder": ["ohe"],
            f"sklearn.preprocessing.{scaler_name}.StandardScaler": ["scaler"],
        }
        assert serialization.name == fixture_name
        self.assertDictEqual(structure, fixture_structure)
        assert new_model.transformer_list[0][0] == fu.transformer_list[0][0]
        assert (
            new_model.transformer_list[0][1].get_params() == fu.transformer_list[0][1].get_params()
        )
        assert new_model.transformer_list[1][0] == fu.transformer_list[1][0]
        assert (
            new_model.transformer_list[1][1].get_params() == fu.transformer_list[1][1].get_params()
        )

        assert [step[0] for step in new_model.transformer_list] == [
            step[0] for step in fu.transformer_list
        ]
        assert new_model.transformer_list[0][1] is not fu.transformer_list[0][1]
        assert new_model.transformer_list[1][1] is not fu.transformer_list[1][1]

        fu.set_params(scaler="drop")
        serialization, new_model = self._serialization_test_helper(
            fu,
            X=self.X,
            y=self.y,
            subcomponent_parameters=("ohe", "transformer_list"),
            dependencies_mock_call_count=(3, 6),
        )
        assert (
            serialization.name == "sklearn.pipeline.FeatureUnion("
            f"ohe=sklearn.preprocessing.{module_name_encoder}.OneHotEncoder,"
            "scaler=drop)"
        )
        assert new_model.transformer_list[1][1] == "drop"

    @pytest.mark.sklearn()
    def test_serialize_feature_union_switched_names(self):
        ohe_params = {"categories": "auto"} if LooseVersion(sklearn.__version__) >= "0.20" else {}
        ohe = sklearn.preprocessing.OneHotEncoder(**ohe_params)
        scaler = sklearn.preprocessing.StandardScaler()
        fu1 = sklearn.pipeline.FeatureUnion(transformer_list=[("ohe", ohe), ("scaler", scaler)])
        fu2 = sklearn.pipeline.FeatureUnion(transformer_list=[("scaler", ohe), ("ohe", scaler)])

        fu1_serialization, _ = self._serialization_test_helper(
            fu1,
            X=None,
            y=None,
            subcomponent_parameters=(),
            dependencies_mock_call_count=(3, 6),
        )
        fu2_serialization, _ = self._serialization_test_helper(
            fu2,
            X=None,
            y=None,
            subcomponent_parameters=(),
            dependencies_mock_call_count=(3, 6),
        )

        # OneHotEncoder was moved to _encoders module in 0.20
        module_name_encoder = "_encoders" if LooseVersion(sklearn.__version__) >= "0.20" else "data"
        scaler_name = "data" if LooseVersion(sklearn.__version__) < "0.22" else "_data"
        assert (
            fu1_serialization.name == "sklearn.pipeline.FeatureUnion("
            f"ohe=sklearn.preprocessing.{module_name_encoder}.OneHotEncoder,"
            f"scaler=sklearn.preprocessing.{scaler_name}.StandardScaler)"
        )
        assert (
            fu2_serialization.name == "sklearn.pipeline.FeatureUnion("
            f"scaler=sklearn.preprocessing.{module_name_encoder}.OneHotEncoder,"
            f"ohe=sklearn.preprocessing.{scaler_name}.StandardScaler)"
        )

    @pytest.mark.sklearn()
    def test_serialize_complex_flow(self):
        ohe = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier(),
        )
        model = sklearn.pipeline.Pipeline(
            steps=[("ohe", ohe), ("scaler", scaler), ("boosting", boosting)],
        )
        parameter_grid = {
            "boosting__base_estimator__max_depth": scipy.stats.randint(1, 10),
            "boosting__learning_rate": scipy.stats.uniform(0.01, 0.99),
            "boosting__n_estimators": [1, 5, 10, 100],
        }
        # convert to ordered dict, sorted by keys) due to param grid check
        parameter_grid = OrderedDict(sorted(parameter_grid.items()))
        cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)
        rs = sklearn.model_selection.RandomizedSearchCV(
            estimator=model,
            param_distributions=parameter_grid,
            cv=cv,
        )
        serialized, new_model = self._serialization_test_helper(
            rs,
            X=self.X,
            y=self.y,
            subcomponent_parameters=(),
            dependencies_mock_call_count=(6, 12),
        )
        structure = serialized.get_structure("name")
        # OneHotEncoder was moved to _encoders module in 0.20
        module_name_encoder = "_encoders" if LooseVersion(sklearn.__version__) >= "0.20" else "data"
        ohe_name = "sklearn.preprocessing.%s.OneHotEncoder" % module_name_encoder
        scaler_name = "sklearn.preprocessing.{}.StandardScaler".format(
            "data" if LooseVersion(sklearn.__version__) < "0.22" else "_data",
        )
        tree_name = "sklearn.tree.{}.DecisionTreeClassifier".format(
            "tree" if LooseVersion(sklearn.__version__) < "0.22" else "_classes",
        )
        weight_name = "weight" if LooseVersion(sklearn.__version__) < "0.22" else "_weight"
        boosting_name = "sklearn.ensemble.{}_boosting.AdaBoostClassifier(base_estimator={})".format(
            weight_name,
            tree_name,
        )
        pipeline_name = "sklearn.pipeline.Pipeline(ohe={},scaler={}," "boosting={})".format(
            ohe_name,
            scaler_name,
            boosting_name,
        )
        fixture_name = (
            "sklearn.model_selection._search.RandomizedSearchCV" "(estimator=%s)" % pipeline_name
        )
        fixture_structure = {
            ohe_name: ["estimator", "ohe"],
            scaler_name: ["estimator", "scaler"],
            tree_name: ["estimator", "boosting", "base_estimator"],
            boosting_name: ["estimator", "boosting"],
            pipeline_name: ["estimator"],
            fixture_name: [],
        }
        assert serialized.name == fixture_name
        assert structure == fixture_structure

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.21",
        reason="Pipeline till 0.20 doesn't support 'passthrough'",
    )
    def test_serialize_strings_as_pipeline_steps(self):
        import sklearn.compose

        # First check: test whether a passthrough in a pipeline is serialized correctly
        model = sklearn.pipeline.Pipeline(steps=[("transformer", "passthrough")])
        serialized = self.extension.model_to_flow(model)
        assert isinstance(serialized, OpenMLFlow)
        assert len(serialized.components) == 1
        assert serialized.components["transformer"].name == "passthrough"
        serialized = self.extension._serialize_sklearn(
            ("transformer", "passthrough"),
            parent_model=model,
        )
        assert serialized == ("transformer", "passthrough")
        extracted_info = self.extension._extract_information_from_model(model)
        assert len(extracted_info[2]) == 1
        assert isinstance(extracted_info[2]["transformer"], OpenMLFlow)
        assert extracted_info[2]["transformer"].name == "passthrough"

        # Second check: test whether a lone passthrough in a column transformer is serialized
        # correctly
        model = sklearn.compose.ColumnTransformer([("passthrough", "passthrough", (0,))])
        serialized = self.extension.model_to_flow(model)
        assert isinstance(serialized, OpenMLFlow)
        assert len(serialized.components) == 1
        assert serialized.components["passthrough"].name == "passthrough"
        serialized = self.extension._serialize_sklearn(
            ("passthrough", "passthrough"),
            parent_model=model,
        )
        assert serialized == ("passthrough", "passthrough")
        extracted_info = self.extension._extract_information_from_model(model)
        assert len(extracted_info[2]) == 1
        assert isinstance(extracted_info[2]["passthrough"], OpenMLFlow)
        assert extracted_info[2]["passthrough"].name == "passthrough"

        # Third check: passthrough and drop in a column transformer
        model = sklearn.compose.ColumnTransformer(
            [("passthrough", "passthrough", (0,)), ("drop", "drop", (1,))],
        )
        serialized = self.extension.model_to_flow(model)
        assert isinstance(serialized, OpenMLFlow)
        assert len(serialized.components) == 2
        assert serialized.components["passthrough"].name == "passthrough"
        assert serialized.components["drop"].name == "drop"
        serialized = self.extension._serialize_sklearn(
            ("passthrough", "passthrough"),
            parent_model=model,
        )
        assert serialized == ("passthrough", "passthrough")
        extracted_info = self.extension._extract_information_from_model(model)
        assert len(extracted_info[2]) == 2
        assert isinstance(extracted_info[2]["passthrough"], OpenMLFlow)
        assert isinstance(extracted_info[2]["drop"], OpenMLFlow)
        assert extracted_info[2]["passthrough"].name == "passthrough"
        assert extracted_info[2]["drop"].name == "drop"

        # Fourth check: having an actual preprocessor in the column transformer, too
        model = sklearn.compose.ColumnTransformer(
            [
                ("passthrough", "passthrough", (0,)),
                ("drop", "drop", (1,)),
                ("test", sklearn.preprocessing.StandardScaler(), (2,)),
            ],
        )
        serialized = self.extension.model_to_flow(model)
        assert isinstance(serialized, OpenMLFlow)
        assert len(serialized.components) == 3
        assert serialized.components["passthrough"].name == "passthrough"
        assert serialized.components["drop"].name == "drop"
        serialized = self.extension._serialize_sklearn(
            ("passthrough", "passthrough"),
            parent_model=model,
        )
        assert serialized == ("passthrough", "passthrough")
        extracted_info = self.extension._extract_information_from_model(model)
        assert len(extracted_info[2]) == 3
        assert isinstance(extracted_info[2]["passthrough"], OpenMLFlow)
        assert isinstance(extracted_info[2]["drop"], OpenMLFlow)
        assert extracted_info[2]["passthrough"].name == "passthrough"
        assert extracted_info[2]["drop"].name == "drop"

        # Fifth check: test whether a lone drop in a feature union is serialized correctly
        model = sklearn.pipeline.FeatureUnion([("drop", "drop")])
        serialized = self.extension.model_to_flow(model)
        assert isinstance(serialized, OpenMLFlow)
        assert len(serialized.components) == 1
        assert serialized.components["drop"].name == "drop"
        serialized = self.extension._serialize_sklearn(("drop", "drop"), parent_model=model)
        assert serialized == ("drop", "drop")
        extracted_info = self.extension._extract_information_from_model(model)
        assert len(extracted_info[2]) == 1
        assert isinstance(extracted_info[2]["drop"], OpenMLFlow)
        assert extracted_info[2]["drop"].name == "drop"

    @pytest.mark.sklearn()
    def test_serialize_type(self):
        supported_types = [float, np.float32, np.float64, int, np.int32, np.int64]
        if LooseVersion(np.__version__) < "1.24":
            supported_types.append(float)
            supported_types.append(int)

        for supported_type in supported_types:
            serialized = self.extension.model_to_flow(supported_type)
            deserialized = self.extension.flow_to_model(serialized)
            assert deserialized == supported_type

    @pytest.mark.sklearn()
    def test_serialize_rvs(self):
        supported_rvs = [
            scipy.stats.norm(loc=1, scale=5),
            scipy.stats.expon(loc=1, scale=5),
            scipy.stats.randint(low=-3, high=15),
        ]

        for supported_rv in supported_rvs:
            serialized = self.extension.model_to_flow(supported_rv)
            deserialized = self.extension.flow_to_model(serialized)
            assert type(deserialized.dist) == type(supported_rv.dist)
            del deserialized.dist
            del supported_rv.dist
            assert deserialized.__dict__ == supported_rv.__dict__

    @pytest.mark.sklearn()
    def test_serialize_function(self):
        serialized = self.extension.model_to_flow(sklearn.feature_selection.chi2)
        deserialized = self.extension.flow_to_model(serialized)
        assert deserialized == sklearn.feature_selection.chi2

    @pytest.mark.sklearn()
    def test_serialize_cvobject(self):
        methods = [sklearn.model_selection.KFold(3), sklearn.model_selection.LeaveOneOut()]
        fixtures = [
            OrderedDict(
                [
                    ("oml-python:serialized_object", "cv_object"),
                    (
                        "value",
                        OrderedDict(
                            [
                                ("name", "sklearn.model_selection._split.KFold"),
                                (
                                    "parameters",
                                    OrderedDict(
                                        [
                                            ("n_splits", "3"),
                                            ("random_state", "null"),
                                            ("shuffle", "false"),
                                        ],
                                    ),
                                ),
                            ],
                        ),
                    ),
                ],
            ),
            OrderedDict(
                [
                    ("oml-python:serialized_object", "cv_object"),
                    (
                        "value",
                        OrderedDict(
                            [
                                ("name", "sklearn.model_selection._split.LeaveOneOut"),
                                ("parameters", OrderedDict()),
                            ],
                        ),
                    ),
                ],
            ),
        ]
        for method, fixture in zip(methods, fixtures):
            m = self.extension.model_to_flow(method)
            assert m == fixture

            m_new = self.extension.flow_to_model(m)
            assert m_new is not m
            assert isinstance(m_new, type(method))

    @pytest.mark.sklearn()
    def test_serialize_simple_parameter_grid(self):
        # We cannot easily test for scipy random variables in here, but they
        # should be covered

        # Examples from the scikit-learn documentation
        models = [sklearn.svm.SVC(), sklearn.ensemble.RandomForestClassifier()]
        grids = [
            [
                OrderedDict([("C", [1, 10, 100, 1000]), ("kernel", ["linear"])]),
                OrderedDict(
                    [("C", [1, 10, 100, 1000]), ("gamma", [0.001, 0.0001]), ("kernel", ["rbf"])],
                ),
            ],
            OrderedDict(
                [
                    ("bootstrap", [True, False]),
                    ("criterion", ["gini", "entropy"]),
                    ("max_depth", [3, None]),
                    ("max_features", [1, 3, 10]),
                    ("min_samples_leaf", [1, 3, 10]),
                    ("min_samples_split", [1, 3, 10]),
                ],
            ),
        ]

        for grid, model in zip(grids, models):
            serialized = self.extension.model_to_flow(grid)
            deserialized = self.extension.flow_to_model(serialized)

            assert deserialized == grid
            assert deserialized is not grid
            # providing error_score because nan != nan
            hpo = sklearn.model_selection.GridSearchCV(
                param_grid=grid,
                estimator=model,
                error_score=-1000,
            )

            serialized = self.extension.model_to_flow(hpo)
            deserialized = self.extension.flow_to_model(serialized)
            assert hpo.param_grid == deserialized.param_grid
            assert hpo.estimator.get_params() == deserialized.estimator.get_params()
            hpo_params = hpo.get_params(deep=False)
            deserialized_params = deserialized.get_params(deep=False)
            del hpo_params["estimator"]
            del deserialized_params["estimator"]
            assert hpo_params == deserialized_params

    @pytest.mark.sklearn()
    @unittest.skip(
        "This feature needs further reworking. If we allow several "
        "components, we need to register them all in the downstream "
        "flows. This is so far not implemented.",
    )
    def test_serialize_advanced_grid(self):
        # TODO instead a GridSearchCV object should be serialized

        # This needs to be in its own function because we cannot simply check
        # for the equality of the grid, because scikit-learn objects don't
        # really support the equality operator
        # This will only work with sklearn==0.18
        N_FEATURES_OPTIONS = [2, 4, 8]
        C_OPTIONS = [1, 10, 100, 1000]
        grid = [
            {
                "reduce_dim": [
                    sklearn.decomposition.PCA(iterated_power=7),
                    sklearn.decomposition.NMF(),
                ],
                "reduce_dim__n_components": N_FEATURES_OPTIONS,
                "classify__C": C_OPTIONS,
            },
            {
                "reduce_dim": [
                    sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2),
                ],
                "reduce_dim__k": N_FEATURES_OPTIONS,
                "classify__C": C_OPTIONS,
            },
        ]

        serialized = self.extension.model_to_flow(grid)
        deserialized = self.extension.flow_to_model(serialized)

        assert (
            grid[0]["reduce_dim"][0].get_params() == deserialized[0]["reduce_dim"][0].get_params()
        )
        assert grid[0]["reduce_dim"][0] is not deserialized[0]["reduce_dim"][0]
        assert (
            grid[0]["reduce_dim"][1].get_params() == deserialized[0]["reduce_dim"][1].get_params()
        )
        assert grid[0]["reduce_dim"][1] is not deserialized[0]["reduce_dim"][1]
        assert grid[0]["reduce_dim__n_components"] == deserialized[0]["reduce_dim__n_components"]
        assert grid[0]["classify__C"] == deserialized[0]["classify__C"]
        assert (
            grid[1]["reduce_dim"][0].get_params() == deserialized[1]["reduce_dim"][0].get_params()
        )
        assert grid[1]["reduce_dim"][0] is not deserialized[1]["reduce_dim"][0]
        assert grid[1]["reduce_dim__k"] == deserialized[1]["reduce_dim__k"]
        assert grid[1]["classify__C"] == deserialized[1]["classify__C"]

    @pytest.mark.sklearn()
    def test_serialize_advanced_grid_fails(self):
        # This unit test is checking that the test we skip above would actually fail

        param_grid = {
            "base_estimator": [
                sklearn.tree.DecisionTreeClassifier(),
                sklearn.tree.ExtraTreeClassifier(),
            ],
        }

        clf = sklearn.model_selection.GridSearchCV(
            sklearn.ensemble.BaggingClassifier(),
            param_grid=param_grid,
        )
        with pytest.raises(
            TypeError,
            match=re.compile(r".*OpenML.*Flow.*is not JSON serializable", flags=re.DOTALL),
        ):
            self.extension.model_to_flow(clf)

    @pytest.mark.sklearn()
    def test_serialize_resampling(self):
        kfold = sklearn.model_selection.StratifiedKFold(n_splits=4, shuffle=True)
        serialized = self.extension.model_to_flow(kfold)
        deserialized = self.extension.flow_to_model(serialized)
        # Best approximation to get_params()
        assert str(deserialized) == str(kfold)
        assert deserialized is not kfold

    @pytest.mark.sklearn()
    def test_hypothetical_parameter_values(self):
        # The hypothetical parameter values of true, 1, 0.1 formatted as a
        # string (and their correct serialization and deserialization) an only
        #  be checked inside a model

        model = Model("true", "1", "0.1")

        serialized = self.extension.model_to_flow(model)
        serialized.external_version = "sklearn==test123"
        deserialized = self.extension.flow_to_model(serialized)
        assert deserialized.get_params() == model.get_params()
        assert deserialized is not model

    @pytest.mark.sklearn()
    def test_gaussian_process(self):
        opt = scipy.optimize.fmin_l_bfgs_b
        kernel = sklearn.gaussian_process.kernels.Matern()
        gp = sklearn.gaussian_process.GaussianProcessClassifier(kernel=kernel, optimizer=opt)
        with pytest.raises(
            TypeError,
            match=r"Matern\(length_scale=1, nu=1.5\), <class 'sklearn.gaussian_process.kernels.Matern'>",
        ):
            self.extension.model_to_flow(gp)

    @pytest.mark.sklearn()
    def test_error_on_adding_component_multiple_times_to_flow(self):
        # this function implicitly checks
        # - openml.flows._check_multiple_occurence_of_component_in_flow()
        pca = sklearn.decomposition.PCA()
        pca2 = sklearn.decomposition.PCA()
        pipeline = sklearn.pipeline.Pipeline((("pca1", pca), ("pca2", pca2)))
        fixture = "Found a second occurence of component .*.PCA when trying to serialize Pipeline"
        with pytest.raises(ValueError, match=fixture):
            self.extension.model_to_flow(pipeline)

        fu = sklearn.pipeline.FeatureUnion((("pca1", pca), ("pca2", pca2)))
        fixture = (
            "Found a second occurence of component .*.PCA when trying " "to serialize FeatureUnion"
        )
        with pytest.raises(ValueError, match=fixture):
            self.extension.model_to_flow(fu)

        fs = sklearn.feature_selection.SelectKBest()
        fu2 = sklearn.pipeline.FeatureUnion((("pca1", pca), ("fs", fs)))
        pipeline2 = sklearn.pipeline.Pipeline((("fu", fu2), ("pca2", pca2)))
        fixture = "Found a second occurence of component .*.PCA when trying to serialize Pipeline"
        with pytest.raises(ValueError, match=fixture):
            self.extension.model_to_flow(pipeline2)

    @pytest.mark.sklearn()
    def test_subflow_version_propagated(self):
        this_directory = os.path.dirname(os.path.abspath(__file__))
        tests_directory = os.path.abspath(os.path.join(this_directory, "..", ".."))
        sys.path.append(tests_directory)
        import tests.test_flows.dummy_learn.dummy_forest

        pca = sklearn.decomposition.PCA()
        dummy = tests.test_flows.dummy_learn.dummy_forest.DummyRegressor()
        pipeline = sklearn.pipeline.Pipeline((("pca", pca), ("dummy", dummy)))
        flow = self.extension.model_to_flow(pipeline)
        # In python2.7, the unit tests work differently on travis-ci; therefore,
        # I put the alternative travis-ci answer here as well. While it has a
        # different value, it is still correct as it is a propagation of the
        # subclasses' module name
        assert flow.external_version == "{},{},{}".format(
            self.extension._format_external_version("openml", openml.__version__),
            self.extension._format_external_version("sklearn", sklearn.__version__),
            self.extension._format_external_version("tests", "0.1"),
        )

    @pytest.mark.sklearn()
    @mock.patch("warnings.warn")
    def test_check_dependencies(self, warnings_mock):
        dependencies = ["sklearn==0.1", "sklearn>=99.99.99", "sklearn>99.99.99"]
        for dependency in dependencies:
            self.assertRaises(ValueError, self.extension._check_dependencies, dependency)

    @pytest.mark.sklearn()
    def test_illegal_parameter_names(self):
        # illegal name: estimators
        clf1 = sklearn.ensemble.VotingClassifier(
            estimators=[
                ("estimators", sklearn.ensemble.RandomForestClassifier()),
                ("whatevs", sklearn.ensemble.ExtraTreesClassifier()),
            ],
        )
        clf2 = sklearn.ensemble.VotingClassifier(
            estimators=[
                ("whatevs", sklearn.ensemble.RandomForestClassifier()),
                ("estimators", sklearn.ensemble.ExtraTreesClassifier()),
            ],
        )
        cases = [clf1, clf2]

        for case in cases:
            self.assertRaises(PyOpenMLError, self.extension.model_to_flow, case)

    @pytest.mark.sklearn()
    def test_paralizable_check(self):
        # using this model should pass the test (if param distribution is
        # legal)
        singlecore_bagging = sklearn.ensemble.BaggingClassifier()
        # using this model should return false (if param distribution is legal)
        multicore_bagging = sklearn.ensemble.BaggingClassifier(n_jobs=5)
        # using this param distribution should raise an exception
        illegal_param_dist = {"base__n_jobs": [-1, 0, 1]}
        # using this param distribution should not raise an exception
        legal_param_dist = {"n_estimators": [2, 3, 4]}

        legal_models = [
            sklearn.ensemble.RandomForestClassifier(),
            sklearn.ensemble.RandomForestClassifier(n_jobs=5),
            sklearn.ensemble.RandomForestClassifier(n_jobs=-1),
            sklearn.pipeline.Pipeline(
                steps=[("bag", sklearn.ensemble.BaggingClassifier(n_jobs=1))],
            ),
            sklearn.pipeline.Pipeline(
                steps=[("bag", sklearn.ensemble.BaggingClassifier(n_jobs=5))],
            ),
            sklearn.pipeline.Pipeline(
                steps=[("bag", sklearn.ensemble.BaggingClassifier(n_jobs=-1))],
            ),
            sklearn.model_selection.GridSearchCV(singlecore_bagging, legal_param_dist),
            sklearn.model_selection.GridSearchCV(multicore_bagging, legal_param_dist),
            sklearn.ensemble.BaggingClassifier(
                n_jobs=-1,
                base_estimator=sklearn.ensemble.RandomForestClassifier(n_jobs=5),
            ),
        ]
        illegal_models = [
            sklearn.model_selection.GridSearchCV(singlecore_bagging, illegal_param_dist),
            sklearn.model_selection.GridSearchCV(multicore_bagging, illegal_param_dist),
        ]

        if LooseVersion(sklearn.__version__) < "0.20":
            has_refit_time = [False, False, False, False, False, False, False, False, False]
        else:
            has_refit_time = [False, False, False, False, False, False, True, True, False]

        X, y = sklearn.datasets.load_iris(return_X_y=True)
        for model, refit_time in zip(legal_models, has_refit_time):
            model.fit(X, y)
            assert refit_time == hasattr(model, "refit_time_")

        for model in illegal_models:
            with pytest.raises(PyOpenMLError):
                self.extension._prevent_optimize_n_jobs(model)

    @pytest.mark.sklearn()
    def test__get_fn_arguments_with_defaults(self):
        sklearn_version = LooseVersion(sklearn.__version__)
        if sklearn_version < "0.19":
            fns = [
                (sklearn.ensemble.RandomForestRegressor.__init__, 15),
                (sklearn.tree.DecisionTreeClassifier.__init__, 12),
                (sklearn.pipeline.Pipeline.__init__, 0),
            ]
        elif sklearn_version < "0.21":
            fns = [
                (sklearn.ensemble.RandomForestRegressor.__init__, 16),
                (sklearn.tree.DecisionTreeClassifier.__init__, 13),
                (sklearn.pipeline.Pipeline.__init__, 1),
            ]
        elif sklearn_version < "0.22":
            fns = [
                (sklearn.ensemble.RandomForestRegressor.__init__, 16),
                (sklearn.tree.DecisionTreeClassifier.__init__, 13),
                (sklearn.pipeline.Pipeline.__init__, 2),
            ]
        elif sklearn_version < "0.23":
            fns = [
                (sklearn.ensemble.RandomForestRegressor.__init__, 18),
                (sklearn.tree.DecisionTreeClassifier.__init__, 14),
                (sklearn.pipeline.Pipeline.__init__, 2),
            ]
        elif sklearn_version < "0.24":
            fns = [
                (sklearn.ensemble.RandomForestRegressor.__init__, 18),
                (sklearn.tree.DecisionTreeClassifier.__init__, 14),
                (sklearn.pipeline.Pipeline.__init__, 2),
            ]
        elif sklearn_version < "1.0":
            fns = [
                (sklearn.ensemble.RandomForestRegressor.__init__, 18),
                (sklearn.tree.DecisionTreeClassifier.__init__, 13),
                (sklearn.pipeline.Pipeline.__init__, 2),
            ]
        else:
            # Tested with 1.0 and 1.1
            fns = [
                (sklearn.ensemble.RandomForestRegressor.__init__, 17),
                (sklearn.tree.DecisionTreeClassifier.__init__, 12),
                (sklearn.pipeline.Pipeline.__init__, 2),
            ]

        for fn, num_params_with_defaults in fns:
            defaults, defaultless = self.extension._get_fn_arguments_with_defaults(fn)
            assert isinstance(defaults, dict)
            assert isinstance(defaultless, set)
            # check whether we have both defaults and defaultless params
            assert len(defaults) == num_params_with_defaults
            assert len(defaultless) > 0
            # check no overlap
            self.assertSetEqual(set(defaults.keys()), set(defaults.keys()) - defaultless)
            self.assertSetEqual(defaultless, defaultless - set(defaults.keys()))

    @pytest.mark.sklearn()
    def test_deserialize_with_defaults(self):
        # used the 'initialize_with_defaults' flag of the deserialization
        # method to return a flow that contains default hyperparameter
        # settings.
        steps = [
            ("Imputer", SimpleImputer()),
            ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder()),
            ("Estimator", sklearn.tree.DecisionTreeClassifier()),
        ]
        pipe_orig = sklearn.pipeline.Pipeline(steps=steps)

        pipe_adjusted = sklearn.clone(pipe_orig)
        if LooseVersion(sklearn.__version__) < "0.23":
            params = {
                "Imputer__strategy": "median",
                "OneHotEncoder__sparse": False,
                "Estimator__min_samples_leaf": 42,
            }
        else:
            params = {
                "Imputer__strategy": "mean",
                "OneHotEncoder__sparse": True,
                "Estimator__min_samples_leaf": 1,
            }
        pipe_adjusted.set_params(**params)
        flow = self.extension.model_to_flow(pipe_adjusted)
        pipe_deserialized = self.extension.flow_to_model(flow, initialize_with_defaults=True)

        # we want to compare pipe_deserialized and pipe_orig. We use the flow
        # equals function for this
        assert_flows_equal(
            self.extension.model_to_flow(pipe_orig),
            self.extension.model_to_flow(pipe_deserialized),
        )

    @pytest.mark.sklearn()
    def test_deserialize_adaboost_with_defaults(self):
        # used the 'initialize_with_defaults' flag of the deserialization
        # method to return a flow that contains default hyperparameter
        # settings.
        steps = [
            ("Imputer", SimpleImputer()),
            ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder()),
            (
                "Estimator",
                sklearn.ensemble.AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier()),
            ),
        ]
        pipe_orig = sklearn.pipeline.Pipeline(steps=steps)

        pipe_adjusted = sklearn.clone(pipe_orig)
        if LooseVersion(sklearn.__version__) < "0.22":
            params = {
                "Imputer__strategy": "median",
                "OneHotEncoder__sparse": False,
                "Estimator__n_estimators": 10,
            }
        else:
            params = {
                "Imputer__strategy": "mean",
                "OneHotEncoder__sparse": True,
                "Estimator__n_estimators": 50,
            }
        pipe_adjusted.set_params(**params)
        flow = self.extension.model_to_flow(pipe_adjusted)
        pipe_deserialized = self.extension.flow_to_model(flow, initialize_with_defaults=True)

        # we want to compare pipe_deserialized and pipe_orig. We use the flow
        # equals function for this
        assert_flows_equal(
            self.extension.model_to_flow(pipe_orig),
            self.extension.model_to_flow(pipe_deserialized),
        )

    @pytest.mark.sklearn()
    def test_deserialize_complex_with_defaults(self):
        # used the 'initialize_with_defaults' flag of the deserialization
        # method to return a flow that contains default hyperparameter
        # settings.
        steps = [
            ("Imputer", SimpleImputer()),
            ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder()),
            (
                "Estimator",
                sklearn.ensemble.AdaBoostClassifier(
                    sklearn.ensemble.BaggingClassifier(
                        sklearn.ensemble.GradientBoostingClassifier(),
                    ),
                ),
            ),
        ]
        pipe_orig = sklearn.pipeline.Pipeline(steps=steps)

        pipe_adjusted = sklearn.clone(pipe_orig)
        impute_strategy = "median" if LooseVersion(sklearn.__version__) < "0.23" else "mean"
        sparse = LooseVersion(sklearn.__version__) >= "0.23"
        estimator_name = (
            "base_estimator" if LooseVersion(sklearn.__version__) < "1.2" else "estimator"
        )
        params = {
            "Imputer__strategy": impute_strategy,
            "OneHotEncoder__sparse": sparse,
            "Estimator__n_estimators": 10,
            f"Estimator__{estimator_name}__n_estimators": 10,
            f"Estimator__{estimator_name}__{estimator_name}__learning_rate": 0.1,
        }

        pipe_adjusted.set_params(**params)
        flow = self.extension.model_to_flow(pipe_adjusted)
        pipe_deserialized = self.extension.flow_to_model(flow, initialize_with_defaults=True)

        # we want to compare pipe_deserialized and pipe_orig. We use the flow
        # equals function for this
        assert_flows_equal(
            self.extension.model_to_flow(pipe_orig),
            self.extension.model_to_flow(pipe_deserialized),
        )

    @pytest.mark.sklearn()
    def test_openml_param_name_to_sklearn(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier(),
        )
        model = sklearn.pipeline.Pipeline(steps=[("scaler", scaler), ("boosting", boosting)])
        flow = self.extension.model_to_flow(model)
        task = openml.tasks.get_task(115)  # diabetes; crossvalidation
        run = openml.runs.run_flow_on_task(flow, task)
        run = run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], run.run_id))
        run = openml.runs.get_run(run.run_id)
        setup = openml.setups.get_setup(run.setup_id)

        # make sure to test enough parameters
        assert len(setup.parameters) > 15

        for parameter in setup.parameters.values():
            sklearn_name = self.extension._openml_param_name_to_sklearn(parameter, flow)

            # test the inverse. Currently, OpenML stores the hyperparameter
            # fullName as flow.name + flow.version + parameter.name on the
            # server (but this behaviour is not documented and might or might
            # not change in the future. Hence, we won't offer this
            # transformation functionality in the main package yet.)
            splitted = sklearn_name.split("__")
            if len(splitted) > 1:  # if len is 1, it is part of root flow
                subflow = flow.get_subflow(splitted[0:-1])
            else:
                subflow = flow
            openml_name = f"{subflow.name}({subflow.version})_{splitted[-1]}"
            assert parameter.full_name == openml_name

    @pytest.mark.sklearn()
    def test_obtain_parameter_values_flow_not_from_server(self):
        model = sklearn.linear_model.LogisticRegression(solver="lbfgs")
        flow = self.extension.model_to_flow(model)
        logistic_name = "logistic" if LooseVersion(sklearn.__version__) < "0.22" else "_logistic"
        msg = f"Flow sklearn.linear_model.{logistic_name}.LogisticRegression has no flow_id!"

        with pytest.raises(ValueError, match=msg):
            self.extension.obtain_parameter_values(flow)

        model = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.linear_model.LogisticRegression(
                solver="lbfgs",
            ),
        )
        flow = self.extension.model_to_flow(model)
        flow.flow_id = 1
        with pytest.raises(ValueError, match=msg):
            self.extension.obtain_parameter_values(flow)

    @pytest.mark.sklearn()
    def test_obtain_parameter_values(self):
        model = sklearn.model_selection.RandomizedSearchCV(
            estimator=sklearn.ensemble.RandomForestClassifier(n_estimators=5),
            param_distributions={
                "max_depth": [3, None],
                "max_features": [1, 2, 3, 4],
                "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
            },
            cv=sklearn.model_selection.StratifiedKFold(n_splits=2, random_state=1, shuffle=True),
            n_iter=5,
        )
        flow = self.extension.model_to_flow(model)
        flow.flow_id = 1
        flow.components["estimator"].flow_id = 2
        parameters = self.extension.obtain_parameter_values(flow)
        for parameter in parameters:
            assert parameter["oml:component"] is not None, parameter
            if parameter["oml:name"] == "n_estimators":
                assert parameter["oml:value"] == "5"
                assert parameter["oml:component"] == 2

    @pytest.mark.sklearn()
    def test_numpy_type_allowed_in_flow(self):
        """Simple numpy types should be serializable."""
        dt = sklearn.tree.DecisionTreeClassifier(
            max_depth=np.float64(3.0),
            min_samples_leaf=np.int32(5),
        )
        self.extension.model_to_flow(dt)

    @pytest.mark.sklearn()
    def test_numpy_array_not_allowed_in_flow(self):
        """Simple numpy arrays should not be serializable."""
        bin = sklearn.preprocessing.MultiLabelBinarizer(classes=np.asarray([1, 2, 3]))
        with pytest.raises(TypeError):
            self.extension.model_to_flow(bin)


class TestSklearnExtensionRunFunctions(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp(n_levels=2)
        self.extension = SklearnExtension()

    ################################################################################################
    # Test methods for performing runs with this extension module

    @pytest.mark.sklearn()
    def test_run_model_on_task(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        # using most_frequent imputer since dataset has mixed types and to keep things simple
        pipe = sklearn.pipeline.Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("dummy", sklearn.dummy.DummyClassifier()),
            ],
        )
        openml.runs.run_model_on_task(pipe, task, dataset_format="array")

    @pytest.mark.sklearn()
    def test_seed_model(self):
        # randomized models that are initialized without seeds, can be seeded
        randomized_clfs = [
            sklearn.ensemble.BaggingClassifier(),
            sklearn.model_selection.RandomizedSearchCV(
                sklearn.ensemble.RandomForestClassifier(),
                {
                    "max_depth": [3, None],
                    "max_features": [1, 2, 3, 4],
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                    "random_state": [-1, 0, 1, 2],
                },
                cv=sklearn.model_selection.StratifiedKFold(n_splits=2, shuffle=True),
            ),
            sklearn.dummy.DummyClassifier(),
        ]

        for idx, clf in enumerate(randomized_clfs):
            const_probe = 42
            all_params = clf.get_params()
            params = [key for key in all_params if key.endswith("random_state")]
            assert len(params) > 0

            # before param value is None
            for param in params:
                assert all_params[param] is None

            # now seed the params
            clf_seeded = self.extension.seed_model(clf, const_probe)
            new_params = clf_seeded.get_params()

            randstate_params = [key for key in new_params if key.endswith("random_state")]

            # afterwards, param value is set
            for param in randstate_params:
                assert isinstance(new_params[param], int)
                assert new_params[param] is not None

            if idx == 1:
                assert clf.cv.random_state == 56422

    @pytest.mark.sklearn()
    def test_seed_model_raises(self):
        # the _set_model_seed_where_none should raise exception if random_state is
        # anything else than an int
        randomized_clfs = [
            sklearn.ensemble.BaggingClassifier(random_state=np.random.RandomState(42)),
            sklearn.dummy.DummyClassifier(random_state="OpenMLIsGreat"),
        ]

        for clf in randomized_clfs:
            with pytest.raises(ValueError):
                self.extension.seed_model(model=clf, seed=42)

    @pytest.mark.sklearn()
    def test_run_model_on_fold_classification_1_array(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation

        X, y = task.get_X_and_y()
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        pipeline = sklearn.pipeline.Pipeline(
            steps=[("imp", SimpleImputer()), ("clf", sklearn.tree.DecisionTreeClassifier())],
        )
        # TODO add some mocking here to actually test the innards of this function, too!
        res = self.extension._run_model_on_fold(
            model=pipeline,
            task=task,
            fold_no=0,
            rep_no=0,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )

        y_hat, y_hat_proba, user_defined_measures, trace = res

        # predictions
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == y_test.shape
        assert isinstance(y_hat_proba, pd.DataFrame)
        assert y_hat_proba.shape == (y_test.shape[0], 6)
        np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1), np.ones(y_test.shape))
        # The class '4' (at index 3) is not present in the training data. We check that the
        # predicted probabilities for that class are zero!
        np.testing.assert_array_almost_equal(
            y_hat_proba.iloc[:, 3].to_numpy(),
            np.zeros(y_test.shape),
        )
        for i in (0, 1, 2, 4, 5):
            assert np.any(y_hat_proba.iloc[:, i].to_numpy() != np.zeros(y_test.shape))

        # check user defined measures
        fold_evaluations: dict[str, dict[int, dict[int, float]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
        for measure in user_defined_measures:
            fold_evaluations[measure][0][0] = user_defined_measures[measure]

        # trace. SGD does not produce any
        assert trace is None

        self._check_fold_timing_evaluations(
            fold_evaluations,
            num_repeats=1,
            num_folds=1,
            task_type=task.task_type_id,
            check_scores=False,
        )

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.21",
        reason="SimpleImputer, ColumnTransformer available only after 0.19 and "
        "Pipeline till 0.20 doesn't support indexing and 'passthrough'",
    )
    def test_run_model_on_fold_classification_1_dataframe(self):
        from sklearn.compose import ColumnTransformer

        task = openml.tasks.get_task(1)  # anneal; crossvalidation

        # diff test_run_model_on_fold_classification_1_array()
        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        # Helper functions to return required columns for ColumnTransformer
        cat_imp = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse=False),
        )
        cont_imp = make_pipeline(CustomImputer(strategy="mean"), StandardScaler())
        ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
        pipeline = sklearn.pipeline.Pipeline(
            steps=[("transform", ct), ("estimator", sklearn.tree.DecisionTreeClassifier())],
        )
        # TODO add some mocking here to actually test the innards of this function, too!
        res = self.extension._run_model_on_fold(
            model=pipeline,
            task=task,
            fold_no=0,
            rep_no=0,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )

        y_hat, y_hat_proba, user_defined_measures, trace = res

        # predictions
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == y_test.shape
        assert isinstance(y_hat_proba, pd.DataFrame)
        assert y_hat_proba.shape == (y_test.shape[0], 6)
        np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1), np.ones(y_test.shape))
        # The class '4' (at index 3) is not present in the training data. We check that the
        # predicted probabilities for that class are zero!
        np.testing.assert_array_almost_equal(
            y_hat_proba.iloc[:, 3].to_numpy(),
            np.zeros(y_test.shape),
        )
        for i in (0, 1, 2, 4, 5):
            assert np.any(y_hat_proba.iloc[:, i].to_numpy() != np.zeros(y_test.shape))

        # check user defined measures
        fold_evaluations: dict[str, dict[int, dict[int, float]]]  = collections.defaultdict(lambda: collections.defaultdict(dict))
        for measure in user_defined_measures:
            fold_evaluations[measure][0][0] = user_defined_measures[measure]

        # trace. SGD does not produce any
        assert trace is None

        self._check_fold_timing_evaluations(
            fold_evaluations,
            num_repeats=1,
            num_folds=1,
            task_type=task.task_type_id,
            check_scores=False,
        )

    @pytest.mark.sklearn()
    def test_run_model_on_fold_classification_2(self):
        task = openml.tasks.get_task(7)  # kr-vs-kp; crossvalidation

        X, y = task.get_X_and_y()
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        pipeline = sklearn.model_selection.GridSearchCV(
            sklearn.tree.DecisionTreeClassifier(),
            {"max_depth": [1, 2]},
        )
        # TODO add some mocking here to actually test the innards of this function, too!
        res = self.extension._run_model_on_fold(
            model=pipeline,
            task=task,
            fold_no=0,
            rep_no=0,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )

        y_hat, y_hat_proba, user_defined_measures, trace = res

        # predictions
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == y_test.shape
        assert isinstance(y_hat_proba, pd.DataFrame)
        assert y_hat_proba.shape == (y_test.shape[0], 2)
        np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1), np.ones(y_test.shape))
        for i in (0, 1):
            assert np.any(y_hat_proba.to_numpy()[:, i] != np.zeros(y_test.shape))

        # check user defined measures
        fold_evaluations: dict[str, dict[int, dict[int, float]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
        for measure in user_defined_measures:
            fold_evaluations[measure][0][0] = user_defined_measures[measure]

        # check that it produced and returned a trace object of the correct length
        assert isinstance(trace, OpenMLRunTrace)
        assert len(trace.trace_iterations) == 2

        self._check_fold_timing_evaluations(
            fold_evaluations,
            num_repeats=1,
            num_folds=1,
            task_type=task.task_type_id,
            check_scores=False,
        )

    @pytest.mark.sklearn()
    def test_run_model_on_fold_classification_3(self):
        class HardNaiveBayes(sklearn.naive_bayes.GaussianNB):
            # class for testing a naive bayes classifier that does not allow soft
            # predictions
            def predict_proba(*args, **kwargs):
                raise AttributeError("predict_proba is not available when " "probability=False")

        # task 1 (test server) is important: it is a task with an unused class
        tasks = [
            1,  # anneal; crossvalidation
            3,  # anneal; crossvalidation
            115,  # diabetes; crossvalidation
        ]
        flow = unittest.mock.Mock()
        flow.name = "dummy"

        for task_id in tasks:
            task = openml.tasks.get_task(task_id)
            X, y = task.get_X_and_y()
            train_indices, test_indices = task.get_train_test_split_indices(
                repeat=0,
                fold=0,
                sample=0,
            )
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            clf1 = sklearn.pipeline.Pipeline(
                steps=[
                    ("imputer", SimpleImputer()),
                    ("estimator", sklearn.naive_bayes.GaussianNB()),
                ],
            )
            clf2 = sklearn.pipeline.Pipeline(
                steps=[("imputer", SimpleImputer()), ("estimator", HardNaiveBayes())],
            )

            pred_1, proba_1, _, _ = self.extension._run_model_on_fold(
                model=clf1,
                task=task,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                fold_no=0,
                rep_no=0,
            )
            pred_2, proba_2, _, _ = self.extension._run_model_on_fold(
                model=clf2,
                task=task,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                fold_no=0,
                rep_no=0,
            )

            # verifies that the predictions are identical
            np.testing.assert_array_equal(pred_1, pred_2)
            np.testing.assert_array_almost_equal(np.sum(proba_1, axis=1), np.ones(X_test.shape[0]))
            # Test that there are predictions other than ones and zeros
            assert np.sum(proba_1.to_numpy() == 0) + np.sum(proba_1.to_numpy() == 1) < X_test.shape[
                0
            ] * len(task.class_labels)

            np.testing.assert_array_almost_equal(np.sum(proba_2, axis=1), np.ones(X_test.shape[0]))
            # Test that there are only ones and zeros predicted
            assert np.sum(proba_2.to_numpy() == 0) + np.sum(
                proba_2.to_numpy() == 1
            ) == X_test.shape[0] * len(task.class_labels)

    @pytest.mark.sklearn()
    @pytest.mark.production()
    def test_run_model_on_fold_regression(self):
        # There aren't any regression tasks on the test server
        openml.config.server = self.production_server
        task = openml.tasks.get_task(2999)

        X, y = task.get_X_and_y()
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        pipeline = sklearn.pipeline.Pipeline(
            steps=[("imp", SimpleImputer()), ("clf", sklearn.tree.DecisionTreeRegressor())],
        )
        # TODO add some mocking here to actually test the innards of this function, too!
        res = self.extension._run_model_on_fold(
            model=pipeline,
            task=task,
            fold_no=0,
            rep_no=0,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )

        y_hat, y_hat_proba, user_defined_measures, trace = res

        # predictions
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == y_test.shape
        assert y_hat_proba is None

        # check user defined measures
        fold_evaluations: dict[str, dict[int, dict[int, float]]]  = collections.defaultdict(lambda: collections.defaultdict(dict))
        for measure in user_defined_measures:
            fold_evaluations[measure][0][0] = user_defined_measures[measure]

        # trace. SGD does not produce any
        assert trace is None

        self._check_fold_timing_evaluations(
            fold_evaluations,
            num_repeats=1,
            num_folds=1,
            task_type=task.task_type_id,
            check_scores=False,
        )

    @pytest.mark.sklearn()
    @pytest.mark.production()
    def test_run_model_on_fold_clustering(self):
        # There aren't any regression tasks on the test server
        openml.config.server = self.production_server
        task = openml.tasks.get_task(126033)

        X = task.get_X(dataset_format="array")

        pipeline = sklearn.pipeline.Pipeline(
            steps=[("imp", SimpleImputer()), ("clf", sklearn.cluster.KMeans())],
        )
        # TODO add some mocking here to actually test the innards of this function, too!
        res = self.extension._run_model_on_fold(
            model=pipeline,
            task=task,
            fold_no=0,
            rep_no=0,
            X_train=X,
        )

        y_hat, y_hat_proba, user_defined_measures, trace = res

        # predictions
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == (X.shape[0],)
        assert y_hat_proba is None

        # check user defined measures
        fold_evaluations: dict[str, dict[int, dict[int, float]]]  = collections.defaultdict(lambda: collections.defaultdict(dict))
        for measure in user_defined_measures:
            fold_evaluations[measure][0][0] = user_defined_measures[measure]

        # trace. SGD does not produce any
        assert trace is None

        self._check_fold_timing_evaluations(
            fold_evaluations,
            num_repeats=1,
            num_folds=1,
            task_type=task.task_type_id,
            check_scores=False,
        )

    @pytest.mark.sklearn()
    def test__extract_trace_data(self):
        param_grid = {
            "hidden_layer_sizes": [[5, 5], [10, 10], [20, 20]],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "learning_rate_init": [0.1, 0.01, 0.001, 0.0001],
            "max_iter": [10, 20, 40, 80],
        }
        num_iters = 10
        task = openml.tasks.get_task(20)  # balance-scale; crossvalidation
        clf = sklearn.model_selection.RandomizedSearchCV(
            sklearn.neural_network.MLPClassifier(),
            param_grid,
            n_iter=num_iters,
        )
        # just run the task on the model (without invoking any fancy extension & openml code)
        train, _ = task.get_train_test_split_indices(0, 0)
        X, y = task.get_X_and_y()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[train], y[train])

        # check num layers of MLP
        assert clf.best_estimator_.hidden_layer_sizes in param_grid["hidden_layer_sizes"]

        trace_list = self.extension._extract_trace_data(clf, rep_no=0, fold_no=0)
        trace = self.extension._obtain_arff_trace(clf, trace_list)

        assert isinstance(trace, OpenMLRunTrace)
        assert isinstance(trace_list, list)
        assert len(trace_list) == num_iters

        for trace_iteration in iter(trace):
            assert trace_iteration.repeat == 0
            assert trace_iteration.fold == 0
            assert trace_iteration.iteration >= 0
            assert trace_iteration.iteration <= num_iters
            assert trace_iteration.setup_string is None
            assert isinstance(trace_iteration.evaluation, float)
            assert np.isfinite(trace_iteration.evaluation)
            assert isinstance(trace_iteration.selected, bool)

            assert len(trace_iteration.parameters) == len(param_grid)
            for param in param_grid:
                # Prepend with the "parameter_" prefix
                param_in_trace = "parameter_%s" % param
                assert param_in_trace in trace_iteration.parameters
                param_value = json.loads(trace_iteration.parameters[param_in_trace])
                assert param_value in param_grid[param]

    @pytest.mark.sklearn()
    def test_trim_flow_name(self):
        import re

        long = """sklearn.pipeline.Pipeline(
                    columntransformer=sklearn.compose._column_transformer.ColumnTransformer(
                        numeric=sklearn.pipeline.Pipeline(
                            SimpleImputer=sklearn.preprocessing.imputation.Imputer,
                            standardscaler=sklearn.preprocessing.data.StandardScaler),
                        nominal=sklearn.pipeline.Pipeline(
                            simpleimputer=sklearn.impute.SimpleImputer,
                            onehotencoder=sklearn.preprocessing._encoders.OneHotEncoder)),
                    variancethreshold=sklearn.feature_selection.variance_threshold.VarianceThreshold,
                    svc=sklearn.svm.classes.SVC)"""
        short = "sklearn.Pipeline(ColumnTransformer,VarianceThreshold,SVC)"
        shorter = "sklearn.Pipeline(...,SVC)"
        long_stripped, _ = re.subn(r"\s", "", long)
        assert short == SklearnExtension.trim_flow_name(long_stripped)
        assert shorter == SklearnExtension.trim_flow_name(long_stripped, extra_trim_length=50)

        long = """sklearn.pipeline.Pipeline(
                    imputation=openmlstudy14.preprocessing.ConditionalImputer,
                    hotencoding=sklearn.preprocessing.data.OneHotEncoder,
                    variencethreshold=sklearn.feature_selection.variance_threshold.VarianceThreshold,
                    classifier=sklearn.ensemble.forest.RandomForestClassifier)"""
        short = "sklearn.Pipeline(ConditionalImputer,OneHotEncoder,VarianceThreshold,RandomForestClassifier)"  # noqa: E501
        long_stripped, _ = re.subn(r"\s", "", long)
        assert short == SklearnExtension.trim_flow_name(long_stripped)

        long = """sklearn.pipeline.Pipeline(
                    SimpleImputer=sklearn.preprocessing.imputation.Imputer,
                    VarianceThreshold=sklearn.feature_selection.variance_threshold.VarianceThreshold, # noqa: E501
                    Estimator=sklearn.model_selection._search.RandomizedSearchCV(
                        estimator=sklearn.tree.tree.DecisionTreeClassifier))"""
        short = (
            "sklearn.Pipeline(Imputer,VarianceThreshold,RandomizedSearchCV(DecisionTreeClassifier))"
        )
        long_stripped, _ = re.subn(r"\s", "", long)
        assert short == SklearnExtension.trim_flow_name(long_stripped)

        long = """sklearn.model_selection._search.RandomizedSearchCV(
                    estimator=sklearn.pipeline.Pipeline(
                        SimpleImputer=sklearn.preprocessing.imputation.Imputer,
                        classifier=sklearn.ensemble.forest.RandomForestClassifier))"""
        short = "sklearn.RandomizedSearchCV(Pipeline(Imputer,RandomForestClassifier))"
        long_stripped, _ = re.subn(r"\s", "", long)
        assert short == SklearnExtension.trim_flow_name(long_stripped)

        long = """sklearn.pipeline.FeatureUnion(
                    pca=sklearn.decomposition.pca.PCA,
                    svd=sklearn.decomposition.truncated_svd.TruncatedSVD)"""
        short = "sklearn.FeatureUnion(PCA,TruncatedSVD)"
        long_stripped, _ = re.subn(r"\s", "", long)
        assert short == SklearnExtension.trim_flow_name(long_stripped)

        long = "sklearn.ensemble.forest.RandomForestClassifier"
        short = "sklearn.RandomForestClassifier"
        assert short == SklearnExtension.trim_flow_name(long)

        assert SklearnExtension.trim_flow_name("weka.IsolationForest") == "weka.IsolationForest"

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.21",
        reason="SimpleImputer, ColumnTransformer available only after 0.19 and "
        "Pipeline till 0.20 doesn't support indexing and 'passthrough'",
    )
    def test_run_on_model_with_empty_steps(self):
        from sklearn.compose import ColumnTransformer

        # testing 'drop', 'passthrough', None as non-actionable sklearn estimators
        dataset = openml.datasets.get_dataset(128)  # iris
        task = openml.tasks.get_task(59)  # mfeat-pixel; crossvalidation

        X, y, categorical_ind, feature_names = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="array",
        )
        categorical_ind = np.array(categorical_ind)
        (cat_idx,) = np.where(categorical_ind)
        (cont_idx,) = np.where(~categorical_ind)

        clf = make_pipeline(
            ColumnTransformer(
                [
                    (
                        "cat",
                        make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()),
                        cat_idx.tolist(),
                    ),
                    (
                        "cont",
                        make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                        cont_idx.tolist(),
                    ),
                ],
            ),
        )

        clf = sklearn.pipeline.Pipeline(
            [
                ("dummystep", "passthrough"),  # adding 'passthrough' as an estimator
                ("prep", clf),
                ("classifier", sklearn.svm.SVC(gamma="auto")),
            ],
        )

        # adding 'drop' to a ColumnTransformer
        if not categorical_ind.any():
            clf[1][0].set_params(cat="drop")
        if not (~categorical_ind).any():
            clf[1][0].set_params(cont="drop")

        # serializing model with non-actionable step
        run, flow = openml.runs.run_model_on_task(model=clf, task=task, return_flow=True)

        assert len(flow.components) == 3
        assert isinstance(flow.components["dummystep"], OpenMLFlow)
        assert flow.components["dummystep"].name == "passthrough"
        assert isinstance(flow.components["classifier"], OpenMLFlow)
        if LooseVersion(sklearn.__version__) < "0.22":
            assert flow.components["classifier"].name == "sklearn.svm.classes.SVC"
        else:
            assert flow.components["classifier"].name == "sklearn.svm._classes.SVC"
        assert isinstance(flow.components["prep"], OpenMLFlow)
        assert flow.components["prep"].class_name == "sklearn.pipeline.Pipeline"
        assert isinstance(flow.components["prep"].components["columntransformer"], OpenMLFlow)
        assert isinstance(
            flow.components["prep"].components["columntransformer"].components["cat"], OpenMLFlow
        )
        assert (
            flow.components["prep"].components["columntransformer"].components["cat"].name == "drop"
        )

        # de-serializing flow to a model with non-actionable step
        model = self.extension.flow_to_model(flow)
        model.fit(X, y)
        assert type(model) == type(clf)
        assert model != clf
        assert len(model.named_steps) == 3
        assert model.named_steps["dummystep"] == "passthrough"

        xml = flow._to_dict()
        new_model = self.extension.flow_to_model(OpenMLFlow._from_dict(xml))

        new_model.fit(X, y)
        assert type(new_model) == type(clf)
        assert new_model != clf
        assert len(new_model.named_steps) == 3
        assert new_model.named_steps["dummystep"] == "passthrough"

    @pytest.mark.sklearn()
    def test_sklearn_serialization_with_none_step(self):
        msg = (
            "Cannot serialize objects of None type. Please use a valid "
            "placeholder for None. Note that empty sklearn estimators can be "
            "replaced with 'drop' or 'passthrough'."
        )
        clf = sklearn.pipeline.Pipeline(
            [("dummystep", None), ("classifier", sklearn.svm.SVC(gamma="auto"))],
        )
        with pytest.raises(ValueError, match=msg):
            self.extension.model_to_flow(clf)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    def test_failed_serialization_of_custom_class(self):
        """Check if any custom class inherited from sklearn expectedly fails serialization"""
        try:
            from sklearn.impute import SimpleImputer
        except ImportError:
            # for lower versions
            from sklearn.preprocessing import Imputer as SimpleImputer

        import sklearn.tree
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        cat_imp = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        cont_imp = make_pipeline(CustomImputer(), StandardScaler())
        ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
        clf = Pipeline(
            steps=[("preprocess", ct), ("estimator", sklearn.tree.DecisionTreeClassifier())],
        )  # build a sklearn classifier

        task = openml.tasks.get_task(253)  # profb; crossvalidation
        try:
            _ = openml.runs.run_model_on_task(clf, task)
        except AttributeError as e:
            if e.args[0] == "module '__main__' has no attribute '__version__'":
                raise AttributeError(e)
            else:
                raise Exception(e)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    def test_setupid_with_column_transformer(self):
        """Test to check if inclusion of ColumnTransformer in a pipleline is treated as a new
        flow each time.
        """
        import sklearn.compose
        from sklearn.svm import SVC

        def column_transformer_pipe(task_id):
            task = openml.tasks.get_task(task_id)
            # make columntransformer
            preprocessor = sklearn.compose.ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), cont),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                ],
            )
            # make pipeline
            clf = SVC(gamma="scale", random_state=1)
            pipe = make_pipeline(preprocessor, clf)
            # run task
            run = openml.runs.run_model_on_task(pipe, task, avoid_duplicate_runs=False)
            run.publish()
            return openml.runs.get_run(run.run_id)

        run1 = column_transformer_pipe(11)  # only categorical
        TestBase._mark_entity_for_removal("run", run1.run_id)
        run2 = column_transformer_pipe(23)  # only numeric
        TestBase._mark_entity_for_removal("run", run2.run_id)
        assert run1.setup_id == run2.setup_id
