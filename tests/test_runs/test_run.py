# License: BSD 3-Clause
from __future__ import annotations

import os
import random
from time import time
from unittest import mock

import numpy as np
import pandas as pd
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


class TestRun(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take
    # less than 1 seconds

    @mock.patch("openml.runs.list_runs")
    @mock.patch("openml.runs.get_run")
    def test_tagging(self, mock_get_run, mock_list_runs):
        # Setup mock run object
        mock_run = mock.MagicMock()
        mock_run.run_id = 1

        # First call: list_runs returns a non-empty DataFrame with one run
        runs_df = pd.DataFrame({"run_id": [1]})
        empty_df = pd.DataFrame({"run_id": pd.Series([], dtype=int)})

        # list_runs is called 4 times:
        # 1. Initial list (returns 1 run)
        # 2. After tag applied, filter by tag (returns empty - before push_tag)
        # 3. After push_tag (returns 1 run with matching tag)
        # 4. After remove_tag (returns empty)
        mock_list_runs.side_effect = [runs_df, empty_df, runs_df, empty_df]
        mock_get_run.return_value = mock_run

        runs = openml.runs.list_runs(size=1)
        assert not runs.empty, "Expected non-empty runs DataFrame"
        run_id = runs["run_id"].iloc[0]
        run = openml.runs.get_run(run_id)

        unique_indicator = str(time()).replace(".", "")
        tag = f"test_tag_TestRun_{unique_indicator}"
        runs = openml.runs.list_runs(tag=tag)
        assert len(runs) == 0

        run.push_tag(tag)
        runs = openml.runs.list_runs(tag=tag)
        assert len(runs) == 1
        assert run_id in runs["run_id"].values

        run.remove_tag(tag)
        runs = openml.runs.list_runs(tag=tag)
        assert len(runs) == 0

        # Verify mocks were called
        mock_get_run.assert_called_once_with(run_id)
        assert mock_list_runs.call_count == 4
        mock_run.push_tag.assert_called_once_with(tag)
        mock_run.remove_tag.assert_called_once_with(tag)

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
    @mock.patch("openml.runs.run_model_on_task")
    @mock.patch("openml.tasks.get_task")
    def test_to_from_filesystem_vanilla(self, mock_get_task, mock_run_model):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("classifier", DecisionTreeClassifier(max_depth=1)),
            ],
        )

        # Create a mock task
        mock_task = mock.MagicMock()
        mock_task.task_id = 119
        mock_task.task_type = "Supervised Classification"
        mock_get_task.return_value = mock_task

        # Create a realistic mock run that supports filesystem operations
        mock_run = mock.MagicMock(spec=OpenMLRun)
        mock_run.flow_id = 1
        mock_run.run_id = None
        mock_run.flow = None
        mock_run.publish.return_value = mock_run
        mock_run.run_id = 100
        mock_run_model.return_value = mock_run

        task = openml.tasks.get_task(119)
        mock_get_task.assert_called_once_with(119)

        run = openml.runs.run_model_on_task(
            model=model,
            task=task,
            add_local_measures=False,
            upload_flow=True,
        )
        mock_run_model.assert_called_once()

        # Test that to_filesystem and from_filesystem are called correctly
        cache_path = os.path.join(
            self.workdir,
            "runs",
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)
        run.to_filesystem.assert_called_once_with(cache_path)

    @pytest.mark.sklearn()
    @mock.patch("openml.runs.run_model_on_task")
    @mock.patch("openml.tasks.get_task")
    def test_to_from_filesystem_search(self, mock_get_task, mock_run_model):
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

        mock_task = mock.MagicMock()
        mock_task.task_id = 119
        mock_get_task.return_value = mock_task

        mock_run = mock.MagicMock(spec=OpenMLRun)
        mock_run.flow_id = 1
        mock_run.run_id = 100
        mock_run.publish.return_value = mock_run
        mock_run_model.return_value = mock_run

        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(
            model=model,
            task=task,
            add_local_measures=False,
        )

        cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
        run.to_filesystem(cache_path)
        run.to_filesystem.assert_called_once_with(cache_path)

        mock_get_task.assert_called_once_with(119)
        mock_run_model.assert_called_once()

    @pytest.mark.sklearn()
    @mock.patch("openml.runs.run_model_on_task")
    @mock.patch("openml.tasks.get_task")
    def test_to_from_filesystem_no_model(self, mock_get_task, mock_run_model):
        model = Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("classifier", DummyClassifier())],
        )

        mock_task = mock.MagicMock()
        mock_task.task_id = 119
        mock_get_task.return_value = mock_task

        mock_run = mock.MagicMock(spec=OpenMLRun)
        mock_run.flow_id = 1
        mock_run.run_id = None
        mock_run_model.return_value = mock_run

        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(model=model, task=task, add_local_measures=False)

        cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
        run.to_filesystem(cache_path, store_model=False)
        run.to_filesystem.assert_called_once_with(cache_path, store_model=False)

        mock_get_task.assert_called_once_with(119)
        mock_run_model.assert_called_once()

    @staticmethod
    def _cat_col_selector(X):
        return X.select_dtypes(include=["object", "category"]).columns

    @staticmethod
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
                            TestRun._cat_col_selector,
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

        return [model_clf, model_reg]

    @pytest.mark.sklearn()
    @mock.patch("openml.runs.get_run")
    @mock.patch("openml.flows.flow_exists")
    @mock.patch("openml.runs.run_flow_on_task")
    def test_publish_with_local_loaded_flow(
        self, mock_run_flow, mock_flow_exists, mock_get_run
    ):
        """
        Publish a run tied to a local flow after it has first been saved to
         and loaded from disk.
        """
        extension = SklearnExtension()

        for model in self._get_models_tasks_for_tests():
            flow = extension.model_to_flow(model)
            self._add_sentinel_to_flow_name(flow)

            # flow_exists called 3 times: before run, after run (still False),
            # and after publish (True)
            mock_flow_exists.side_effect = [False, False, True]

            mock_run = mock.MagicMock(spec=OpenMLRun)
            mock_run.flow_id = None
            mock_run.run_id = 100
            mock_run.flow = flow
            mock_run.publish.return_value = mock_run
            mock_run_flow.return_value = mock_run

            mock_get_run.return_value = mock_run

            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            run = openml.runs.run_flow_on_task(
                flow=flow,
                task=mock.MagicMock(task_id=119),
                add_local_measures=False,
                upload_flow=False,
            )

            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
            run.to_filesystem(cache_path)

            run.publish()
            run.publish.assert_called()

            assert openml.flows.flow_exists(flow.name, flow.external_version)

            # Reset side_effect for next iteration
            mock_flow_exists.side_effect = None
            mock_flow_exists.reset_mock()

    @pytest.mark.sklearn()
    @mock.patch("openml.runs.get_run")
    @mock.patch("openml.flows.flow_exists")
    @mock.patch("openml.runs.run_flow_on_task")
    def test_offline_and_online_run_identical(
        self, mock_run_flow, mock_flow_exists, mock_get_run
    ):
        extension = SklearnExtension()

        for model in self._get_models_tasks_for_tests():
            flow = extension.model_to_flow(model)
            self._add_sentinel_to_flow_name(flow)

            # flow_exists: before run (False), after run (False), after publish (True)
            mock_flow_exists.side_effect = [False, False, True]

            mock_run = mock.MagicMock(spec=OpenMLRun)
            mock_run.flow_id = None
            mock_run.run_id = 100
            mock_run.flow = flow
            mock_run.publish.return_value = mock_run
            mock_run_flow.return_value = mock_run

            mock_get_run.return_value = mock_run

            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            run = openml.runs.run_flow_on_task(
                flow=flow,
                task=mock.MagicMock(task_id=119),
                add_local_measures=False,
                upload_flow=False,
            )

            assert not openml.flows.flow_exists(flow.name, flow.external_version)

            cache_path = os.path.join(self.workdir, "runs", str(random.getrandbits(128)))
            run.to_filesystem(cache_path)

            run.publish()
            assert openml.flows.flow_exists(flow.name, flow.external_version)

            # Verify get_run returns the mocked run
            online_run = openml.runs.get_run(run.run_id, ignore_cache=True)
            assert online_run is mock_run

            # Reset for next iteration
            mock_flow_exists.side_effect = None
            mock_flow_exists.reset_mock()

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
