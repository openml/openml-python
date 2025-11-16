# License: BSD 3-Clause
from __future__ import annotations

import os
import random
from time import time

import numpy as np
import pytest
import xmltodict
from openml_sklearn import SklearnExtension
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import openml
from openml import OpenMLRun
from openml.testing import SimpleImputer, TestBase


pytestmark = pytest.mark.usefixtures("with_test_cache", "with_server")


@pytest.fixture
def extension():
    return SklearnExtension()


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


def test_tagging():
        runs = openml.runs.list_runs(size=1)
        assert not runs.empty, "Test server state is incorrect"
        run_id = runs["run_id"].iloc[0]
        run = openml.runs.get_run(run_id)
        # tags can be at most 64 alphanumeric (+ underscore) chars
        unique_indicator = str(time()).replace(".", "")
        tag = f"test_tag_TestRun_{unique_indicator}"
        runs = openml.runs.list_runs(tag=tag)
        assert len(runs) == 0
        run.push_tag(tag)
        runs = openml.runs.list_runs(tag=tag)
        assert len(runs) == 1
        assert run_id in runs["run_id"]
        run.remove_tag(tag)
        runs = openml.runs.list_runs(tag=tag)
        assert len(runs) == 0


def _test_prediction_data_equal(run, run_prime):
    # Determine which attributes are numeric and which not
    num_cols = np.array(
        [d_type == "NUMERIC" for _, d_type in run._generate_arff_dict()["attributes"]],
    )
    # Get run data consistently
    #   (For run from server, .data_content does not exist)
    run_data_content = run.predictions.values
    run_prime_data_content = run_prime.predictions.values

    # Assert numeric and string parts separately
    numeric_part = np.array(run_data_content[:, num_cols], dtype=float)
    numeric_part_prime = np.array(run_prime_data_content[:, num_cols], dtype=float)
    string_part = run_data_content[:, ~num_cols]
    string_part_prime = run_prime_data_content[:, ~num_cols]
    np.testing.assert_array_almost_equal(numeric_part, numeric_part_prime)
    np.testing.assert_array_equal(string_part, string_part_prime)


def _test_run_obj_equals(run, run_prime):
    for dictionary in ["evaluations", "fold_evaluations", "sample_evaluations"]:
        if getattr(run, dictionary) is not None:
            assert getattr(run, dictionary) == getattr(run_prime, dictionary)
        else:
            # should be none or empty
            other = getattr(run_prime, dictionary)
            if other is not None:
                assert other == {}
    assert run._to_xml() == run_prime._to_xml()
    _test_prediction_data_equal(run, run_prime)


@pytest.mark.sklearn()
def test_to_from_filesystem_vanilla(workdir):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("classifier", DecisionTreeClassifier(max_depth=1)),
        ],
    )
    task = openml.tasks.get_task(119)  # diabetes; crossvalidation
    run = openml.runs.run_model_on_task(
        model=model,
        task=task,
        add_local_measures=False,
        upload_flow=True,
    )

    cache_path = os.path.join(
        workdir,
        "runs",
        str(random.getrandbits(128)),
    )
    run.to_filesystem(cache_path)

    run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
    # The flow has been uploaded to server, so only the reference flow_id should be present
    assert run_prime.flow_id is not None
    assert run_prime.flow is None
    _test_run_obj_equals(run, run_prime)


@pytest.mark.sklearn()
@pytest.mark.flaky()
def test_to_from_filesystem_search(workdir):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("classifier", DecisionTreeClassifier(max_depth=1)),
        ],
    )
    model = GridSearchCV(
        estimator=model,
        param_grid={
            "classifier__max_depth": [1, 2, 3, 4, 5],
            "imputer__strategy": ["mean", "median"],
        },
    )

    task = openml.tasks.get_task(119)  # diabetes; crossvalidation
    run = openml.runs.run_model_on_task(
        model=model,
        task=task,
        add_local_measures=False,
    )

    cache_path = os.path.join(workdir, "runs", str(random.getrandbits(128)))
    run.to_filesystem(cache_path)

    run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
    _test_run_obj_equals(run, run_prime)

@pytest.mark.sklearn()
def test_to_from_filesystem_no_model(workdir):
    model = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("classifier", DummyClassifier())],
    )
    task = openml.tasks.get_task(119)  # diabetes; crossvalidation
    run = openml.runs.run_model_on_task(model=model, task=task, add_local_measures=False)

    cache_path = os.path.join(workdir, "runs", str(random.getrandbits(128)))
    run.to_filesystem(cache_path, store_model=False)
    # obtain run from filesystem
    openml.runs.OpenMLRun.from_filesystem(cache_path, expect_model=False)
    # assert default behaviour is throwing an error
    with pytest.raises(ValueError, match="Could not find model.pkl"):
        openml.runs.OpenMLRun.from_filesystem(cache_path)


def _cat_col_selector(X):
    return X.select_dtypes(include=["object", "category"]).columns


def _get_sentinel(sentinel=None):
    """Get a unique test sentinel."""
    if sentinel is None:
        sentinel = str(random.getrandbits(128))
    if not sentinel.startswith("TEST"):
        sentinel = f"TEST{sentinel}"
    return sentinel


def _add_sentinel_to_flow_name(flow, sentinel=None):
    """Add test sentinel to flow name to avoid conflicts."""
    sentinel = _get_sentinel(sentinel=sentinel)
    flows_to_visit = [flow]
    while len(flows_to_visit) > 0:
        current_flow = flows_to_visit.pop()
        current_flow.name = f"{sentinel}{current_flow.name}"
        for subflow in current_flow.components.values():
            flows_to_visit.append(subflow)
    return flow, sentinel


def _get_models_tasks_for_tests():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    basic_preprocessing = [
        (
            "cat_handling",
            ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        _cat_col_selector,
                    )
                ],
                remainder="passthrough",
            ),
        ),
        ("imp", SimpleImputer()),
    ]
    model_clf = Pipeline(
        [
            *basic_preprocessing,
            ("classifier", DummyClassifier(strategy="prior")),
        ],
    )
    model_reg = Pipeline(
        [
            *basic_preprocessing,
            (
                "regressor",
                # LR because dummy does not produce enough float-like values
                LinearRegression(),
            ),
        ],
    )

    task_clf = openml.tasks.get_task(119)  # diabetes; hold out validation
    task_reg = openml.tasks.get_task(733)  # quake; crossvalidation
    return [(model_clf, task_clf), (model_reg, task_reg)]


def assert_run_prediction_data(task, run, model):
        # -- Get y_pred and y_true as it should be stored in the run
        n_repeats, n_folds, n_samples = task.get_split_dimensions()
        if (n_repeats > 1) or (n_samples > 1):
            raise ValueError("Test does not support this task type's split dimensions.")

        X, y = task.get_X_and_y()

        # Check correctness of y_true and y_pred in run
        for fold_id in range(n_folds):
            # Get data for fold
            _, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold_id, sample=0)
            train_mask = np.full(len(X), True)
            train_mask[test_indices] = False

            # Get train / test
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[~train_mask]
            y_test = y[~train_mask]

            # Get y_pred
            y_pred = model.fit(X_train, y_train).predict(X_test)

            # Get stored data for fold
            saved_fold_data = run.predictions[run.predictions["fold"] == fold_id].sort_values(
                by="row_id",
            )
            saved_y_pred = saved_fold_data["prediction"].values
            gt_key = "truth" if "truth" in list(saved_fold_data) else "correct"
            saved_y_test = saved_fold_data[gt_key].values

            assert_method = np.testing.assert_array_almost_equal
            if task.task_type == "Supervised Classification":
                assert_method = np.testing.assert_array_equal
            y_test = y_test.values

            # Assert correctness
            assert_method(y_pred, saved_y_pred)
            assert_method(y_test, saved_y_test)


@pytest.mark.sklearn()
def test_publish_with_local_loaded_flow(workdir, extension):
    """
    Publish a run tied to a local flow after it has first been saved to
     and loaded from disk.
    """
    for model, task in _get_models_tasks_for_tests():
        # Make sure the flow does not exist on the server yet.
        flow = extension.model_to_flow(model)
        _add_sentinel_to_flow_name(flow)
        assert not openml.flows.flow_exists(flow.name, flow.external_version)

        run = openml.runs.run_flow_on_task(
            flow=flow,
            task=task,
            add_local_measures=False,
            upload_flow=False,
        )

        # Make sure that the flow has not been uploaded as requested.
        assert not openml.flows.flow_exists(flow.name, flow.external_version)

        # Make sure that the prediction data stored in the run is correct.
        assert_run_prediction_data(task, run, clone(model))

        cache_path = os.path.join(workdir, "runs", str(random.getrandbits(128)))
        run.to_filesystem(cache_path)
        # obtain run from filesystem
        loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)
        loaded_run.publish()

@pytest.mark.sklearn()
def test_offline_and_online_run_identical(workdir, extension):
    for model, task in _get_models_tasks_for_tests():
        # Make sure the flow does not exist on the server yet.
        flow = extension.model_to_flow(model)
        _add_sentinel_to_flow_name(flow)
        assert not openml.flows.flow_exists(flow.name, flow.external_version)

        run = openml.runs.run_flow_on_task(
            flow=flow,
            task=task,
            add_local_measures=False,
            upload_flow=False,
        )

        # Make sure that the flow has not been uploaded as requested.
        assert not openml.flows.flow_exists(flow.name, flow.external_version)

        # Load from filesystem
        cache_path = os.path.join(workdir, "runs", str(random.getrandbits(128)))
        run.to_filesystem(cache_path)
        loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)

        # Assert identical for offline - offline
        _test_run_obj_equals(run, loaded_run)

        # Publish and test for offline - online
        run.publish()
        assert openml.flows.flow_exists(flow.name, flow.external_version)

        try:
            online_run = openml.runs.get_run(run.run_id, ignore_cache=True)
            _test_prediction_data_equal(run, online_run)
        finally:
            pass  # No cleanup in pytest version

def test_run_setup_string_included_in_xml():
    SETUP_STRING = "setup-string"
    run = OpenMLRun(
        task_id=0,
        flow_id=None,  # if not none, flow parameters are required.
        dataset_id=0,
        setup_string=SETUP_STRING,
    )
    xml = run._to_xml()
    run_dict = xmltodict.parse(xml)["oml:run"]
    assert "oml:setup_string" in run_dict
    assert run_dict["oml:setup_string"] == SETUP_STRING

    recreated_run = openml.runs.functions._create_run_from_xml(xml, from_server=False)
    assert recreated_run.setup_string == SETUP_STRING
