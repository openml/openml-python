# License: BSD 3-Clause
from __future__ import annotations

import os
import random
from time import time

import numpy as np
import pytest
import xmltodict
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import openml
import openml.extensions.sklearn
from openml import OpenMLRun
from openml.testing import SimpleImputer, TestBase


class TestRun(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take
    # less than 1 seconds

    def test_tagging(self):
        runs = openml.runs.list_runs(size=1, output_format="dataframe")
        assert not runs.empty, "Test server state is incorrect"
        run_id = runs["run_id"].iloc[0]
        run = openml.runs.get_run(run_id)
        # tags can be at most 64 alphanumeric (+ underscore) chars
        unique_indicator = str(time()).replace(".", "")
        tag = f"test_tag_TestRun_{unique_indicator}"
        runs = openml.runs.list_runs(tag=tag, output_format="dataframe")
        assert len(runs) == 0
        run.push_tag(tag)
        runs = openml.runs.list_runs(tag=tag, output_format="dataframe")
        assert len(runs) == 1
        assert run_id in runs["run_id"]
        run.remove_tag(tag)
        runs = openml.runs.list_runs(tag=tag, output_format="dataframe")
        assert len(runs) == 0

    @staticmethod
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

    def _test_run_obj_equals(self, run, run_prime):
        for dictionary in ["evaluations", "fold_evaluations", "sample_evaluations"]:
            if getattr(run, dictionary) is not None:
                self.assertDictEqual(getattr(run, dictionary), getattr(run_prime, dictionary))
            else:
                # should be none or empty
                other = getattr(run_prime, dictionary)
                if other is not None:
                    self.assertDictEqual(other, {})
        assert run._to_xml() == run_prime._to_xml()
        self._test_prediction_data_equal(run, run_prime)

        # Test trace
        run_trace_content = run.trace.trace_to_arff()["data"] if run.trace is not None else None

        if run_prime.trace is not None:
            run_prime_trace_content = run_prime.trace.trace_to_arff()["data"]
        else:
            run_prime_trace_content = None

        if run_trace_content is not None:

            def _check_array(array, type_):
                for line in array:
                    for entry in line:
                        assert isinstance(entry, type_)

            int_part = [line[:3] for line in run_trace_content]
            _check_array(int_part, int)
            int_part_prime = [line[:3] for line in run_prime_trace_content]
            _check_array(int_part_prime, int)

            float_part = np.array(
                np.array(run_trace_content)[:, 3:4],
                dtype=float,
            )
            float_part_prime = np.array(
                np.array(run_prime_trace_content)[:, 3:4],
                dtype=float,
            )
            bool_part = [line[4] for line in run_trace_content]
            bool_part_prime = [line[4] for line in run_prime_trace_content]
            for bp, bpp in zip(bool_part, bool_part_prime):
                assert bp in ["true", "false"]
                assert bpp in ["true", "false"]
            string_part = np.array(run_trace_content)[:, 5:]
            string_part_prime = np.array(run_prime_trace_content)[:, 5:]

            np.testing.assert_array_almost_equal(int_part, int_part_prime)
            np.testing.assert_array_almost_equal(float_part, float_part_prime)
            assert bool_part == bool_part_prime
            np.testing.assert_array_equal(string_part, string_part_prime)
        else:
            assert run_prime_trace_content is None

    @pytest.mark.sklearn()
    def test_to_from_filesystem_vanilla(self):
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
            avoid_duplicate_runs=False,
            upload_flow=True,
        )

        cache_path = os.path.join(
            self.workdir,
            "runs",
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        # The flow has been uploaded to server, so only the reference flow_id should be present
        assert run_prime.flow_id is not None
        assert run_prime.flow is None
        self._test_run_obj_equals(run, run_prime)
        run_prime.publish()
        TestBase._mark_entity_for_removal("run", run_prime.run_id)
        TestBase.logger.info(
            "collected from {}: {}".format(__file__.split("/")[-1], run_prime.run_id),
        )

    @pytest.mark.sklearn()
    @pytest.mark.flaky()
    def test_to_from_filesystem_search(self):
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
            avoid_duplicate_runs=False,
        )

        cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        self._test_run_obj_equals(run, run_prime)
        run_prime.publish()
        TestBase._mark_entity_for_removal("run", run_prime.run_id)
        TestBase.logger.info(
            "collected from {}: {}".format(__file__.split("/")[-1], run_prime.run_id),
        )

    @pytest.mark.sklearn()
    def test_to_from_filesystem_no_model(self):
        model = Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("classifier", DummyClassifier())],
        )
        task = openml.tasks.get_task(119)  # diabetes; crossvalidation
        run = openml.runs.run_model_on_task(model=model, task=task, add_local_measures=False)

        cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
        run.to_filesystem(cache_path, store_model=False)
        # obtain run from filesystem
        openml.runs.OpenMLRun.from_filesystem(cache_path, expect_model=False)
        # assert default behaviour is throwing an error
        with self.assertRaises(ValueError, msg="Could not find model.pkl"):
            openml.runs.OpenMLRun.from_filesystem(cache_path)

    @staticmethod
    def _get_models_tasks_for_tests():
        model_clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("classifier", DummyClassifier(strategy="prior")),
            ],
        )
        model_reg = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
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

    @staticmethod
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
                y_pred = np.take(task.class_labels, y_pred)
                y_test = np.take(task.class_labels, y_test)
                assert_method = np.testing.assert_array_equal

            # Assert correctness
            assert_method(y_pred, saved_y_pred)
            assert_method(y_test, saved_y_test)

    @pytest.mark.sklearn()
    def test_publish_with_local_loaded_flow(self):
        """
        Publish a run tied to a local flow after it has first been saved to
         and loaded from disk.
        """
        extension = openml.extensions.sklearn.SklearnExtension()

        for model, task in self._get_models_tasks_for_tests():
            # Make sure the flow does not exist on the server yet.
            flow = extension.model_to_flow(model)
            self._add_sentinel_to_flow_name(flow)
            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            run = openml.runs.run_flow_on_task(
                flow=flow,
                task=task,
                add_local_measures=False,
                avoid_duplicate_runs=False,
                upload_flow=False,
            )

            # Make sure that the flow has not been uploaded as requested.
            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            # Make sure that the prediction data stored in the run is correct.
            self.assert_run_prediction_data(task, run, clone(model))

            cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
            run.to_filesystem(cache_path)
            # obtain run from filesystem
            loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)
            loaded_run.publish()

            # Clean up
            TestBase._mark_entity_for_removal("run", loaded_run.run_id)
            TestBase.logger.info(
                "collected from {}: {}".format(__file__.split("/")[-1], loaded_run.run_id),
            )

            # make sure the flow is published as part of publishing the run.
            assert openml.flows.flow_exists(flow.name, flow.external_version)
            openml.runs.get_run(loaded_run.run_id)

    @pytest.mark.sklearn()
    def test_offline_and_online_run_identical(self):
        extension = openml.extensions.sklearn.SklearnExtension()

        for model, task in self._get_models_tasks_for_tests():
            # Make sure the flow does not exist on the server yet.
            flow = extension.model_to_flow(model)
            self._add_sentinel_to_flow_name(flow)
            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            run = openml.runs.run_flow_on_task(
                flow=flow,
                task=task,
                add_local_measures=False,
                avoid_duplicate_runs=False,
                upload_flow=False,
            )

            # Make sure that the flow has not been uploaded as requested.
            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            # Load from filesystem
            cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
            run.to_filesystem(cache_path)
            loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)

            # Assert identical for offline - offline
            self._test_run_obj_equals(run, loaded_run)

            # Publish and test for offline - online
            run.publish()
            assert openml.flows.flow_exists(flow.name, flow.external_version)

            try:
                online_run = openml.runs.get_run(run.run_id, ignore_cache=True)
                self._test_prediction_data_equal(run, online_run)
            finally:
                # Clean up
                TestBase._mark_entity_for_removal("run", run.run_id)
                TestBase.logger.info(
                    "collected from {}: {}".format(__file__.split("/")[-1], loaded_run.run_id),
                )

    def test_run_setup_string_included_in_xml(self):
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
