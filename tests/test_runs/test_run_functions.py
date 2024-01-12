# License: BSD 3-Clause
from __future__ import annotations

import ast
import os
import random
import time
import unittest
import warnings
from distutils.version import LooseVersion
from unittest import mock

import arff
import joblib
import numpy as np
import pandas as pd
import pytest
import requests
import sklearn
from joblib import parallel_backend
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import openml
import openml._api_calls
import openml.exceptions
import openml.extensions.sklearn
from openml.exceptions import (
    OpenMLNotAuthorizedError,
    OpenMLServerException,
)
from openml.extensions.sklearn import cat, cont
from openml.runs.functions import (
    _run_task_get_arffcontent,
    delete_run,
    format_prediction,
    run_exists,
)
from openml.runs.trace import OpenMLRunTrace
from openml.tasks import TaskType
from openml.testing import (
    CustomImputer,
    SimpleImputer,
    TestBase,
    check_task_existence,
    create_request_response,
)


class TestRun(TestBase):
    _multiprocess_can_split_ = True
    TEST_SERVER_TASK_MISSING_VALS = {
        "task_id": 96,
        "n_missing_vals": 67,
        "n_test_obs": 227,
        "nominal_indices": [0, 3, 4, 5, 6, 8, 9, 11, 12],
        "numeric_indices": [1, 2, 7, 10, 13, 14],
        "task_meta_data": {
            "task_type": TaskType.SUPERVISED_CLASSIFICATION,
            "dataset_id": 16,  # credit-a
            "estimation_procedure_id": 1,
            "target_name": "class",
        },
    }
    TEST_SERVER_TASK_SIMPLE = {
        "task_id": 119,
        "n_missing_vals": 0,
        "n_test_obs": 253,
        "nominal_indices": [],
        "numeric_indices": [*range(8)],
        "task_meta_data": {
            "task_type": TaskType.SUPERVISED_CLASSIFICATION,
            "dataset_id": 20,  # diabetes
            "estimation_procedure_id": 1,
            "target_name": "class",
        },
    }
    TEST_SERVER_TASK_REGRESSION = {
        "task_id": 1605,
        "n_missing_vals": 0,
        "n_test_obs": 2178,
        "nominal_indices": [],
        "numeric_indices": [*range(8)],
        "task_meta_data": {
            "task_type": TaskType.SUPERVISED_REGRESSION,
            "dataset_id": 123,  # quake
            "estimation_procedure_id": 7,
            "target_name": "richter",
        },
    }

    # Suppress warnings to facilitate testing
    hide_warnings = True
    if hide_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def setUp(self):
        super().setUp()
        self.extension = openml.extensions.sklearn.SklearnExtension()

    def _wait_for_processed_run(self, run_id, max_waiting_time_seconds):
        # it can take a while for a run to be processed on the OpenML (test)
        # server however, sometimes it is good to wait (a bit) for this, to
        # properly test a function. In this case, we wait for max_waiting_time_
        # seconds on this to happen, probing the server every 10 seconds to
        # speed up the process

        # time.time() works in seconds
        start_time = time.time()
        while time.time() - start_time < max_waiting_time_seconds:
            run = openml.runs.get_run(run_id, ignore_cache=True)

            try:
                openml.runs.get_run_trace(run_id)
            except openml.exceptions.OpenMLServerException:
                time.sleep(10)
                continue

            if len(run.evaluations) == 0:
                time.sleep(10)
                continue

            return

        raise RuntimeError(
            f"Could not find any evaluations! Please check whether run {run_id} was "
            "evaluated correctly on the server",
        )

    def _assert_predictions_equal(self, predictions, predictions_prime):
        assert np.array(predictions_prime["data"]).shape == np.array(predictions["data"]).shape

        # The original search model does not submit confidence
        # bounds, so we can not compare the arff line
        compare_slice = [0, 1, 2, -1, -2]
        for idx in range(len(predictions["data"])):
            # depends on the assumption "predictions are in same order"
            # that does not necessarily hold.
            # But with the current code base, it holds.
            for col_idx in compare_slice:
                val_1 = predictions["data"][idx][col_idx]
                val_2 = predictions_prime["data"][idx][col_idx]
                if isinstance(val_1, float) or isinstance(val_2, float):
                    self.assertAlmostEqual(
                        float(val_1),
                        float(val_2),
                        places=6,
                    )
                else:
                    assert val_1 == val_2

    def _rerun_model_and_compare_predictions(self, run_id, model_prime, seed, create_task_obj):
        run = openml.runs.get_run(run_id)

        # TODO: assert holdout task

        # downloads the predictions of the old task
        file_id = run.output_files["predictions"]
        predictions_url = openml._api_calls._file_id_to_url(file_id)
        response = openml._api_calls._download_text_file(predictions_url)
        predictions = arff.loads(response)

        # if create_task_obj=False, task argument in run_model_on_task is specified task_id
        if create_task_obj:
            task = openml.tasks.get_task(run.task_id)
            run_prime = openml.runs.run_model_on_task(
                model=model_prime,
                task=task,
                avoid_duplicate_runs=False,
                seed=seed,
            )
        else:
            run_prime = openml.runs.run_model_on_task(
                model=model_prime,
                task=run.task_id,
                avoid_duplicate_runs=False,
                seed=seed,
            )

        predictions_prime = run_prime._generate_arff_dict()

        self._assert_predictions_equal(predictions, predictions_prime)
        pd.testing.assert_frame_equal(
            run.predictions,
            run_prime.predictions,
            check_dtype=False,  # Loaded ARFF reads NUMERIC as float, even if integer.
        )

    def _perform_run(
        self,
        task_id,
        num_instances,
        n_missing_vals,
        clf,
        flow_expected_rsv=None,
        seed=1,
        check_setup=True,
        sentinel=None,
    ):
        """
        Runs a classifier on a task, and performs some basic checks.
        Also uploads the run.

        Parameters
        ----------
        task_id : int

        num_instances: int
            The expected length of the prediction file (number of test
            instances in original dataset)

        n_missing_values: int

        clf: sklearn.base.BaseEstimator
            The classifier to run

        flow_expected_rsv: str
            The expected random state value for the flow (check by hand,
            depends on seed parameter)

        seed: int
            The seed with which the RSV for runs will be initialized

        check_setup: bool
            If set to True, the flow will be downloaded again and
            reinstantiated, for consistency with original flow.

        sentinel: optional, str
            in case the sentinel should be user specified

        Returns
        -------
        run: OpenMLRun
            The performed run (with run id)
        """
        classes_without_random_state = [
            "sklearn.model_selection._search.GridSearchCV",
            "sklearn.pipeline.Pipeline",
        ]
        if LooseVersion(sklearn.__version__) < "0.22":
            classes_without_random_state.append("sklearn.linear_model.base.LinearRegression")
        else:
            classes_without_random_state.append("sklearn.linear_model._base.LinearRegression")

        def _remove_random_state(flow):
            if "random_state" in flow.parameters:
                del flow.parameters["random_state"]
            for component in flow.components.values():
                _remove_random_state(component)

        flow = self.extension.model_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, sentinel)
        if not openml.flows.flow_exists(flow.name, flow.external_version):
            flow.publish()
            TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
            TestBase.logger.info(f"collected from test_run_functions: {flow.flow_id}")

        task = openml.tasks.get_task(task_id)

        X, y = task.get_X_and_y()
        assert np.count_nonzero(np.isnan(X)) == n_missing_vals
        run = openml.runs.run_flow_on_task(
            flow=flow,
            task=task,
            seed=seed,
            avoid_duplicate_runs=openml.config.avoid_duplicate_runs,
        )
        run_ = run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info(f"collected from test_run_functions: {run.run_id}")
        assert run_ == run
        assert isinstance(run.dataset_id, int)

        # This is only a smoke check right now
        # TODO add a few asserts here
        run._to_xml()
        if run.trace is not None:
            # This is only a smoke check right now
            # TODO add a few asserts here
            run.trace.trace_to_arff()

        # check arff output
        assert len(run.data_content) == num_instances

        if check_setup:
            # test the initialize setup function
            run_id = run_.run_id
            run_server = openml.runs.get_run(run_id)
            clf_server = openml.setups.initialize_model(
                setup_id=run_server.setup_id,
            )
            flow_local = self.extension.model_to_flow(clf)
            flow_server = self.extension.model_to_flow(clf_server)

            if flow.class_name not in classes_without_random_state:
                error_msg = "Flow class %s (id=%d) does not have a random " "state parameter" % (
                    flow.class_name,
                    flow.flow_id,
                )
                assert "random_state" in flow.parameters, error_msg
                # If the flow is initialized from a model without a random
                # state, the flow is on the server without any random state
                assert flow.parameters["random_state"] == "null"
                # As soon as a flow is run, a random state is set in the model.
                # If a flow is re-instantiated
                assert flow_local.parameters["random_state"] == flow_expected_rsv
                assert flow_server.parameters["random_state"] == flow_expected_rsv
            _remove_random_state(flow_local)
            _remove_random_state(flow_server)
            openml.flows.assert_flows_equal(flow_local, flow_server)

            # and test the initialize setup from run function
            clf_server2 = openml.runs.initialize_model_from_run(
                run_id=run_server.run_id,
            )
            flow_server2 = self.extension.model_to_flow(clf_server2)
            if flow.class_name not in classes_without_random_state:
                assert flow_server2.parameters["random_state"] == flow_expected_rsv

            _remove_random_state(flow_server2)
            openml.flows.assert_flows_equal(flow_local, flow_server2)

            # self.assertEqual(clf.get_params(), clf_prime.get_params())
            # self.assertEqual(clf, clf_prime)

        downloaded = openml.runs.get_run(run_.run_id)
        assert "openml-python" in downloaded.tags

        # TODO make sure that these attributes are instantiated when
        # downloading a run? Or make sure that the trace object is created when
        # running a flow on a task (and not only the arff object is created,
        # so that the two objects can actually be compared):
        # downloaded_run_trace = downloaded._generate_trace_arff_dict()
        # self.assertEqual(run_trace, downloaded_run_trace)
        return run

    def _check_sample_evaluations(
        self,
        sample_evaluations,
        num_repeats,
        num_folds,
        num_samples,
        max_time_allowed=60000,
    ):
        """
        Checks whether the right timing measures are attached to the run
        (before upload). Test is only performed for versions >= Python3.3

        In case of check_n_jobs(clf) == false, please do not perform this
        check (check this condition outside of this function. )
        default max_time_allowed (per fold, in milli seconds) = 1 minute,
        quite pessimistic
        """
        # a dict mapping from openml measure to a tuple with the minimum and
        # maximum allowed value
        check_measures = {
            # should take at least one millisecond (?)
            "usercpu_time_millis_testing": (0, max_time_allowed),
            "usercpu_time_millis_training": (0, max_time_allowed),
            "usercpu_time_millis": (0, max_time_allowed),
            "wall_clock_time_millis_training": (0, max_time_allowed),
            "wall_clock_time_millis_testing": (0, max_time_allowed),
            "wall_clock_time_millis": (0, max_time_allowed),
            "predictive_accuracy": (0, 1),
        }

        assert isinstance(sample_evaluations, dict)
        assert set(sample_evaluations.keys()) == set(check_measures.keys())

        for measure in check_measures:
            if measure in sample_evaluations:
                num_rep_entrees = len(sample_evaluations[measure])
                assert num_rep_entrees == num_repeats
                for rep in range(num_rep_entrees):
                    num_fold_entrees = len(sample_evaluations[measure][rep])
                    assert num_fold_entrees == num_folds
                    for fold in range(num_fold_entrees):
                        num_sample_entrees = len(sample_evaluations[measure][rep][fold])
                        assert num_sample_entrees == num_samples
                        for sample in range(num_sample_entrees):
                            evaluation = sample_evaluations[measure][rep][fold][sample]
                            assert isinstance(evaluation, float)
                            if not (os.environ.get("CI_WINDOWS") or os.name == "nt"):
                                # Windows seems to get an eval-time of 0 sometimes.
                                assert evaluation > 0
                            assert evaluation < max_time_allowed

    @pytest.mark.sklearn()
    def test_run_regression_on_classif_task(self):
        task_id = 115  # diabetes; crossvalidation

        clf = LinearRegression()
        task = openml.tasks.get_task(task_id)
        # internally dataframe is loaded and targets are categorical
        # which LinearRegression() cannot handle
        with pytest.raises(
            AttributeError, match="'LinearRegression' object has no attribute 'classes_'"
        ):
            openml.runs.run_model_on_task(
                model=clf,
                task=task,
                avoid_duplicate_runs=False,
                dataset_format="array",
            )

    @pytest.mark.sklearn()
    def test_check_erronous_sklearn_flow_fails(self):
        task_id = 115  # diabetes; crossvalidation
        task = openml.tasks.get_task(task_id)

        # Invalid parameter values
        clf = LogisticRegression(C="abc", solver="lbfgs")
        # The exact error message depends on scikit-learn version.
        # Because the sklearn-extension module is to be separated,
        # I will simply relax specifics of the raised Error.
        # old: r"Penalty term must be positive; got \(C=u?'abc'\)"
        # new: sklearn.utils._param_validation.InvalidParameterError:
        #   The 'C' parameter of LogisticRegression must be a float in the range (0, inf]. Got 'abc' instead.  # noqa: E501
        try:
            from sklearn.utils._param_validation import InvalidParameterError

            exceptions = (ValueError, InvalidParameterError)
        except ImportError:
            exceptions = (ValueError,)
        with pytest.raises(exceptions):
            openml.runs.run_model_on_task(
                task=task,
                model=clf,
            )

    ###########################################################################
    # These unit tests are meant to test the following functions, using a
    # variety of flows:
    # - openml.runs.run_task()
    # - openml.runs.OpenMLRun.publish()
    # - openml.runs.initialize_model()
    # - [implicitly] openml.setups.initialize_model()
    # - openml.runs.initialize_model_from_trace()
    # They're split among several actual functions to allow for parallel
    # execution of the unit tests without the need to add an additional module
    # like unittest2

    def _run_and_upload(
        self,
        clf,
        task_id,
        n_missing_vals,
        n_test_obs,
        flow_expected_rsv,
        num_folds=1,
        num_iterations=5,
        seed=1,
        metric=sklearn.metrics.accuracy_score,
        metric_name="predictive_accuracy",
        task_type=TaskType.SUPERVISED_CLASSIFICATION,
        sentinel=None,
    ):
        def determine_grid_size(param_grid):
            if isinstance(param_grid, dict):
                grid_iterations = 1
                for param in param_grid:
                    grid_iterations *= len(param_grid[param])
                return grid_iterations
            elif isinstance(param_grid, list):
                grid_iterations = 0
                for sub_grid in param_grid:
                    grid_iterations += determine_grid_size(sub_grid)
                return grid_iterations
            else:
                raise TypeError("Param Grid should be of type list " "(GridSearch only) or dict")

        run = self._perform_run(
            task_id,
            n_test_obs,
            n_missing_vals,
            clf,
            flow_expected_rsv=flow_expected_rsv,
            seed=seed,
            sentinel=sentinel,
        )

        # obtain scores using get_metric_score:
        scores = run.get_metric_fn(metric)
        # compare with the scores in user defined measures
        scores_provided = []
        for rep in run.fold_evaluations[metric_name]:
            for fold in run.fold_evaluations[metric_name][rep]:
                scores_provided.append(run.fold_evaluations[metric_name][rep][fold])
        assert sum(scores_provided) == sum(scores)

        if isinstance(clf, BaseSearchCV):
            trace_content = run.trace.trace_to_arff()["data"]
            if isinstance(clf, GridSearchCV):
                grid_iterations = determine_grid_size(clf.param_grid)
                assert len(trace_content) == grid_iterations * num_folds
            else:
                assert len(trace_content) == num_iterations * num_folds

            # downloads the best model based on the optimization trace
            # suboptimal (slow), and not guaranteed to work if evaluation
            # engine is behind.
            # TODO: mock this? We have the arff already on the server
            self._wait_for_processed_run(run.run_id, 600)
            try:
                model_prime = openml.runs.initialize_model_from_trace(
                    run_id=run.run_id,
                    repeat=0,
                    fold=0,
                )
            except openml.exceptions.OpenMLServerException as e:
                e.message = "%s; run_id %d" % (e.message, run.run_id)
                raise e

            self._rerun_model_and_compare_predictions(
                run.run_id,
                model_prime,
                seed,
                create_task_obj=True,
            )
            self._rerun_model_and_compare_predictions(
                run.run_id,
                model_prime,
                seed,
                create_task_obj=False,
            )
        else:
            run_downloaded = openml.runs.get_run(run.run_id)
            sid = run_downloaded.setup_id
            model_prime = openml.setups.initialize_model(sid)
            self._rerun_model_and_compare_predictions(
                run.run_id,
                model_prime,
                seed,
                create_task_obj=True,
            )
            self._rerun_model_and_compare_predictions(
                run.run_id,
                model_prime,
                seed,
                create_task_obj=False,
            )

        # todo: check if runtime is present
        self._check_fold_timing_evaluations(
            fold_evaluations=run.fold_evaluations,
            num_repeats=1,
            num_folds=num_folds,
            task_type=task_type
        )

        # Check if run string and print representation do not run into an error
        #   The above check already verifies that all columns needed for supported
        #   representations are present.
        #   Supported: SUPERVISED_CLASSIFICATION, LEARNING_CURVE, SUPERVISED_REGRESSION
        str(run)
        self.logger.info(run)

        return run

    def _run_and_upload_classification(
        self,
        clf,
        task_id,
        n_missing_vals,
        n_test_obs,
        flow_expected_rsv,
        sentinel=None,
    ):
        num_folds = 1  # because of holdout
        num_iterations = 5  # for base search algorithms
        metric = sklearn.metrics.accuracy_score  # metric class
        metric_name = "predictive_accuracy"  # openml metric name
        task_type = TaskType.SUPERVISED_CLASSIFICATION  # task type

        return self._run_and_upload(
            clf=clf,
            task_id=task_id,
            n_missing_vals=n_missing_vals,
            n_test_obs=n_test_obs,
            flow_expected_rsv=flow_expected_rsv,
            num_folds=num_folds,
            num_iterations=num_iterations,
            metric=metric,
            metric_name=metric_name,
            task_type=task_type,
            sentinel=sentinel,
        )

    def _run_and_upload_regression(
        self,
        clf,
        task_id,
        n_missing_vals,
        n_test_obs,
        flow_expected_rsv,
        sentinel=None,
    ):
        num_folds = 10  # because of cross-validation
        num_iterations = 5  # for base search algorithms
        metric = sklearn.metrics.mean_absolute_error  # metric class
        metric_name = "mean_absolute_error"  # openml metric name
        task_type = TaskType.SUPERVISED_REGRESSION  # task type

        return self._run_and_upload(
            clf=clf,
            task_id=task_id,
            n_missing_vals=n_missing_vals,
            n_test_obs=n_test_obs,
            flow_expected_rsv=flow_expected_rsv,
            num_folds=num_folds,
            num_iterations=num_iterations,
            metric=metric,
            metric_name=metric_name,
            task_type=task_type,
            sentinel=sentinel,
        )

    @pytest.mark.sklearn()
    def test_run_and_upload_logistic_regression(self):
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        task_id = self.TEST_SERVER_TASK_SIMPLE["task_id"]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE["n_missing_vals"]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE["n_test_obs"]
        self._run_and_upload_classification(lr, task_id, n_missing_vals, n_test_obs, "62501")

    @pytest.mark.sklearn()
    def test_run_and_upload_linear_regression(self):
        lr = LinearRegression()
        task_id = self.TEST_SERVER_TASK_REGRESSION["task_id"]

        task_meta_data = self.TEST_SERVER_TASK_REGRESSION["task_meta_data"]
        _task_id = check_task_existence(**task_meta_data)
        if _task_id is not None:
            task_id = _task_id
        else:
            new_task = openml.tasks.create_task(**task_meta_data)
            # publishes the new task
            try:
                new_task = new_task.publish()
                task_id = new_task.task_id
            except OpenMLServerException as e:
                if e.code == 614:  # Task already exists
                    # the exception message contains the task_id that was matched in the format
                    # 'Task already exists. - matched id(s): [xxxx]'
                    task_id = ast.literal_eval(e.message.split("matched id(s):")[-1].strip())[0]
                else:
                    raise Exception(repr(e))
            # mark to remove the uploaded task
            TestBase._mark_entity_for_removal("task", task_id)
            TestBase.logger.info(f"collected from test_run_functions: {task_id}")

        n_missing_vals = self.TEST_SERVER_TASK_REGRESSION["n_missing_vals"]
        n_test_obs = self.TEST_SERVER_TASK_REGRESSION["n_test_obs"]
        self._run_and_upload_regression(lr, task_id, n_missing_vals, n_test_obs, "62501")

    @pytest.mark.sklearn()
    def test_run_and_upload_pipeline_dummy_pipeline(self):
        pipeline1 = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("dummy", DummyClassifier(strategy="prior")),
            ],
        )
        task_id = self.TEST_SERVER_TASK_SIMPLE["task_id"]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE["n_missing_vals"]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE["n_test_obs"]
        self._run_and_upload_classification(pipeline1, task_id, n_missing_vals, n_test_obs, "62501")

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    def test_run_and_upload_column_transformer_pipeline(self):
        import sklearn.compose
        import sklearn.impute

        def get_ct_cf(nominal_indices, numeric_indices):
            inner = sklearn.compose.ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        make_pipeline(
                            SimpleImputer(strategy="mean"),
                            sklearn.preprocessing.StandardScaler(),
                        ),
                        numeric_indices,
                    ),
                    (
                        "nominal",
                        make_pipeline(
                            CustomImputer(strategy="most_frequent"),
                            sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
                        ),
                        nominal_indices,
                    ),
                ],
                remainder="passthrough",
            )
            return sklearn.pipeline.Pipeline(
                steps=[
                    ("transformer", inner),
                    ("classifier", sklearn.tree.DecisionTreeClassifier()),
                ],
            )

        sentinel = self._get_sentinel()
        self._run_and_upload_classification(
            get_ct_cf(
                self.TEST_SERVER_TASK_SIMPLE["nominal_indices"],
                self.TEST_SERVER_TASK_SIMPLE["numeric_indices"],
            ),
            self.TEST_SERVER_TASK_SIMPLE["task_id"],
            self.TEST_SERVER_TASK_SIMPLE["n_missing_vals"],
            self.TEST_SERVER_TASK_SIMPLE["n_test_obs"],
            "62501",
            sentinel=sentinel,
        )
        # Due to #602, it is important to test this model on two tasks
        # with different column specifications
        self._run_and_upload_classification(
            get_ct_cf(
                self.TEST_SERVER_TASK_MISSING_VALS["nominal_indices"],
                self.TEST_SERVER_TASK_MISSING_VALS["numeric_indices"],
            ),
            self.TEST_SERVER_TASK_MISSING_VALS["task_id"],
            self.TEST_SERVER_TASK_MISSING_VALS["n_missing_vals"],
            self.TEST_SERVER_TASK_MISSING_VALS["n_test_obs"],
            "62501",
            sentinel=sentinel,
        )

    @pytest.mark.sklearn()
    @unittest.skip("https://github.com/openml/OpenML/issues/1180")
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    @mock.patch("warnings.warn")
    def test_run_and_upload_knn_pipeline(self, warnings_mock):
        cat_imp = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        cont_imp = make_pipeline(CustomImputer(), StandardScaler())
        from sklearn.compose import ColumnTransformer
        from sklearn.neighbors import KNeighborsClassifier

        ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
        pipeline2 = Pipeline(
            steps=[
                ("Imputer", ct),
                ("VarianceThreshold", VarianceThreshold()),
                (
                    "Estimator",
                    RandomizedSearchCV(
                        KNeighborsClassifier(),
                        {"n_neighbors": list(range(2, 10))},
                        cv=3,
                        n_iter=10,
                    ),
                ),
            ],
        )

        task_id = self.TEST_SERVER_TASK_MISSING_VALS["task_id"]
        n_missing_vals = self.TEST_SERVER_TASK_MISSING_VALS["n_missing_vals"]
        n_test_obs = self.TEST_SERVER_TASK_MISSING_VALS["n_test_obs"]
        self._run_and_upload_classification(pipeline2, task_id, n_missing_vals, n_test_obs, "62501")
        # The warning raised is:
        # "The total space of parameters 8 is smaller than n_iter=10.
        # Running 8 iterations. For exhaustive searches, use GridSearchCV."
        # It is raised three times because we once run the model to upload something and then run
        # it again twice to compare that the predictions are reproducible.
        warning_msg = (
            "The total space of parameters 8 is smaller than n_iter=10. "
            "Running 8 iterations. For exhaustive searches, use GridSearchCV."
        )
        call_count = 0
        for _warnings in warnings_mock.call_args_list:
            if _warnings[0][0] == warning_msg:
                call_count += 1
        assert call_count == 3

    @pytest.mark.sklearn()
    def test_run_and_upload_gridsearch(self):
        gridsearch = GridSearchCV(
            BaggingClassifier(base_estimator=SVC()),
            {"base_estimator__C": [0.01, 0.1, 10], "base_estimator__gamma": [0.01, 0.1, 10]},
            cv=3,
        )
        task_id = self.TEST_SERVER_TASK_SIMPLE["task_id"]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE["n_missing_vals"]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE["n_test_obs"]
        run = self._run_and_upload_classification(
            clf=gridsearch,
            task_id=task_id,
            n_missing_vals=n_missing_vals,
            n_test_obs=n_test_obs,
            flow_expected_rsv="62501",
        )
        assert len(run.trace.trace_iterations) == 9

    @pytest.mark.sklearn()
    def test_run_and_upload_randomsearch(self):
        randomsearch = RandomizedSearchCV(
            RandomForestClassifier(n_estimators=5),
            {
                "max_depth": [3, None],
                "max_features": [1, 2, 3, 4],
                "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
            },
            cv=StratifiedKFold(n_splits=2, shuffle=True),
            n_iter=5,
        )
        # The random states for the RandomizedSearchCV is set after the
        # random state of the RandomForestClassifier is set, therefore,
        # it has a different value than the other examples before
        task_id = self.TEST_SERVER_TASK_SIMPLE["task_id"]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE["n_missing_vals"]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE["n_test_obs"]
        run = self._run_and_upload_classification(
            clf=randomsearch,
            task_id=task_id,
            n_missing_vals=n_missing_vals,
            n_test_obs=n_test_obs,
            flow_expected_rsv="12172",
        )
        assert len(run.trace.trace_iterations) == 5
        trace = openml.runs.get_run_trace(run.run_id)
        assert len(trace.trace_iterations) == 5

    @pytest.mark.sklearn()
    def test_run_and_upload_maskedarrays(self):
        # This testcase is important for 2 reasons:
        # 1) it verifies the correct handling of masked arrays (not all
        # parameters are active)
        # 2) it verifies the correct handling of a 2-layered grid search
        gridsearch = GridSearchCV(
            RandomForestClassifier(n_estimators=5),
            [{"max_features": [2, 4]}, {"min_samples_leaf": [1, 10]}],
            cv=StratifiedKFold(n_splits=2, shuffle=True),
        )
        # The random states for the GridSearchCV is set after the
        # random state of the RandomForestClassifier is set, therefore,
        # it has a different value than the other examples before
        task_id = self.TEST_SERVER_TASK_SIMPLE["task_id"]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE["n_missing_vals"]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE["n_test_obs"]
        self._run_and_upload_classification(
            gridsearch,
            task_id,
            n_missing_vals,
            n_test_obs,
            "12172",
        )

    ##########################################################################

    @pytest.mark.sklearn()
    def test_learning_curve_task_1(self):
        task_id = 801  # diabates dataset
        num_test_instances = 6144  # for learning curve
        num_missing_vals = 0
        num_repeats = 1
        num_folds = 10
        num_samples = 8

        pipeline1 = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("dummy", DummyClassifier(strategy="prior")),
            ],
        )
        run = self._perform_run(
            task_id,
            num_test_instances,
            num_missing_vals,
            pipeline1,
            flow_expected_rsv="62501",
        )
        self._check_sample_evaluations(run.sample_evaluations, num_repeats, num_folds, num_samples)

    @pytest.mark.sklearn()
    def test_learning_curve_task_2(self):
        task_id = 801  # diabates dataset
        num_test_instances = 6144  # for learning curve
        num_missing_vals = 0
        num_repeats = 1
        num_folds = 10
        num_samples = 8

        pipeline2 = Pipeline(
            steps=[
                ("Imputer", SimpleImputer(strategy="median")),
                ("VarianceThreshold", VarianceThreshold()),
                (
                    "Estimator",
                    RandomizedSearchCV(
                        DecisionTreeClassifier(),
                        {
                            "min_samples_split": [2**x for x in range(1, 8)],
                            "min_samples_leaf": [2**x for x in range(7)],
                        },
                        cv=3,
                        n_iter=10,
                    ),
                ),
            ],
        )
        run = self._perform_run(
            task_id,
            num_test_instances,
            num_missing_vals,
            pipeline2,
            flow_expected_rsv="62501",
        )
        self._check_sample_evaluations(run.sample_evaluations, num_repeats, num_folds, num_samples)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.21",
        reason="Pipelines don't support indexing (used for the assert check)",
    )
    def test_initialize_cv_from_run(self):
        randomsearch = Pipeline(
            [
                ("enc", OneHotEncoder(handle_unknown="ignore")),
                (
                    "rs",
                    RandomizedSearchCV(
                        RandomForestClassifier(n_estimators=5),
                        {
                            "max_depth": [3, None],
                            "max_features": [1, 2, 3, 4],
                            "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                            "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            "bootstrap": [True, False],
                            "criterion": ["gini", "entropy"],
                        },
                        cv=StratifiedKFold(n_splits=2, shuffle=True),
                        n_iter=2,
                    ),
                ),
            ],
        )

        task = openml.tasks.get_task(11)  # kr-vs-kp; holdout
        run = openml.runs.run_model_on_task(
            model=randomsearch,
            task=task,
            avoid_duplicate_runs=False,
            seed=1,
        )
        run_ = run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info(f"collected from test_run_functions: {run.run_id}")
        run = openml.runs.get_run(run_.run_id)

        modelR = openml.runs.initialize_model_from_run(run_id=run.run_id)
        modelS = openml.setups.initialize_model(setup_id=run.setup_id)

        assert modelS[-1].cv.random_state == 62501
        assert modelR[-1].cv.random_state == 62501

    def _test_local_evaluations(self, run):
        # compare with the scores in user defined measures
        accuracy_scores_provided = []
        for rep in run.fold_evaluations["predictive_accuracy"]:
            for fold in run.fold_evaluations["predictive_accuracy"][rep]:
                accuracy_scores_provided.append(
                    run.fold_evaluations["predictive_accuracy"][rep][fold],
                )
        accuracy_scores = run.get_metric_fn(sklearn.metrics.accuracy_score)
        np.testing.assert_array_almost_equal(accuracy_scores_provided, accuracy_scores)

        # also check if we can obtain some other scores:
        tests = [
            (sklearn.metrics.cohen_kappa_score, {"weights": None}),
            (sklearn.metrics.roc_auc_score, {}),
            (sklearn.metrics.average_precision_score, {}),
            (sklearn.metrics.precision_score, {"average": "macro"}),
            (sklearn.metrics.brier_score_loss, {}),
        ]
        if LooseVersion(sklearn.__version__) < "0.23":
            tests.append((sklearn.metrics.jaccard_similarity_score, {}))
        else:
            tests.append((sklearn.metrics.jaccard_score, {}))
        for _test_idx, test in enumerate(tests):
            alt_scores = run.get_metric_fn(
                sklearn_fn=test[0],
                kwargs=test[1],
            )
            assert len(alt_scores) == 10
            for idx in range(len(alt_scores)):
                assert alt_scores[idx] >= 0
                assert alt_scores[idx] <= 1

    @pytest.mark.sklearn()
    def test_local_run_swapped_parameter_order_model(self):
        clf = DecisionTreeClassifier()
        australian_task = 595  # Australian; crossvalidation
        task = openml.tasks.get_task(australian_task)

        # task and clf are purposely in the old order
        run = openml.runs.run_model_on_task(
            task,
            clf,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        self._test_local_evaluations(run)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="SimpleImputer doesn't handle mixed type DataFrame as input",
    )
    def test_local_run_swapped_parameter_order_flow(self):
        # construct sci-kit learn classifier
        clf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("estimator", RandomForestClassifier(n_estimators=10)),
            ],
        )

        flow = self.extension.model_to_flow(clf)
        # download task
        task = openml.tasks.get_task(7)  # kr-vs-kp; crossvalidation

        # invoke OpenML run
        run = openml.runs.run_flow_on_task(
            task,
            flow,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        self._test_local_evaluations(run)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="SimpleImputer doesn't handle mixed type DataFrame as input",
    )
    def test_local_run_metric_score(self):
        # construct sci-kit learn classifier
        clf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("estimator", RandomForestClassifier(n_estimators=10)),
            ],
        )

        # download task
        task = openml.tasks.get_task(7)  # kr-vs-kp; crossvalidation

        # invoke OpenML run
        run = openml.runs.run_model_on_task(
            model=clf,
            task=task,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        self._test_local_evaluations(run)

    @pytest.mark.production()
    def test_online_run_metric_score(self):
        openml.config.server = self.production_server

        # important to use binary classification task,
        # due to assertions
        run = openml.runs.get_run(9864498)

        self._test_local_evaluations(run)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="SimpleImputer doesn't handle mixed type DataFrame as input",
    )
    def test_initialize_model_from_run(self):
        clf = sklearn.pipeline.Pipeline(
            steps=[
                ("Imputer", SimpleImputer(strategy="most_frequent")),
                ("VarianceThreshold", VarianceThreshold(threshold=0.05)),
                ("Estimator", GaussianNB()),
            ],
        )
        task_meta_data = {
            "task_type": TaskType.SUPERVISED_CLASSIFICATION,
            "dataset_id": 128,  # iris
            "estimation_procedure_id": 1,
            "target_name": "class",
        }
        _task_id = check_task_existence(**task_meta_data)
        if _task_id is not None:
            task_id = _task_id
        else:
            new_task = openml.tasks.create_task(**task_meta_data)
            # publishes the new task
            try:
                new_task = new_task.publish()
                task_id = new_task.task_id
            except OpenMLServerException as e:
                if e.code == 614:  # Task already exists
                    # the exception message contains the task_id that was matched in the format
                    # 'Task already exists. - matched id(s): [xxxx]'
                    task_id = ast.literal_eval(e.message.split("matched id(s):")[-1].strip())[0]
                else:
                    raise Exception(repr(e))
            # mark to remove the uploaded task
            TestBase._mark_entity_for_removal("task", task_id)
            TestBase.logger.info(f"collected from test_run_functions: {task_id}")

        task = openml.tasks.get_task(task_id)
        run = openml.runs.run_model_on_task(
            model=clf,
            task=task,
            avoid_duplicate_runs=False,
        )
        run_ = run.publish()
        TestBase._mark_entity_for_removal("run", run_.run_id)
        TestBase.logger.info(f"collected from test_run_functions: {run_.run_id}")
        run = openml.runs.get_run(run_.run_id)

        modelR = openml.runs.initialize_model_from_run(run_id=run.run_id)
        modelS = openml.setups.initialize_model(setup_id=run.setup_id)

        flowR = self.extension.model_to_flow(modelR)
        flowS = self.extension.model_to_flow(modelS)
        flowL = self.extension.model_to_flow(clf)
        openml.flows.assert_flows_equal(flowR, flowL)
        openml.flows.assert_flows_equal(flowS, flowL)

        assert flowS.components["Imputer"].parameters["strategy"] == '"most_frequent"'
        assert flowS.components["VarianceThreshold"].parameters["threshold"] == "0.05"

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="SimpleImputer doesn't handle mixed type DataFrame as input",
    )
    def test__run_exists(self):
        # would be better to not sentinel these clfs,
        # so we do not have to perform the actual runs
        # and can just check their status on line
        rs = 1
        clfs = [
            sklearn.pipeline.Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="mean")),
                    ("VarianceThreshold", VarianceThreshold(threshold=0.05)),
                    ("Estimator", DecisionTreeClassifier(max_depth=4)),
                ],
            ),
            sklearn.pipeline.Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("VarianceThreshold", VarianceThreshold(threshold=0.1)),
                    ("Estimator", DecisionTreeClassifier(max_depth=4)),
                ],
            ),
        ]

        task = openml.tasks.get_task(115)  # diabetes; crossvalidation

        for clf in clfs:
            try:
                # first populate the server with this run.
                # skip run if it was already performed.
                run = openml.runs.run_model_on_task(
                    model=clf,
                    task=task,
                    seed=rs,
                    avoid_duplicate_runs=True,
                    upload_flow=True,
                )
                run.publish()
                TestBase._mark_entity_for_removal("run", run.run_id)
                TestBase.logger.info(f"collected from test_run_functions: {run.run_id}")
            except openml.exceptions.PyOpenMLError:
                # run already existed. Great.
                pass

            flow = self.extension.model_to_flow(clf)
            flow_exists = openml.flows.flow_exists(flow.name, flow.external_version)
            assert flow_exists > 0, "Server says flow from run does not exist."
            # Do NOT use get_flow reinitialization, this potentially sets
            # hyperparameter values wrong. Rather use the local model.
            downloaded_flow = openml.flows.get_flow(flow_exists)
            downloaded_flow.model = clf
            setup_exists = openml.setups.setup_exists(downloaded_flow)
            assert setup_exists > 0, "Server says setup of run does not exist."
            run_ids = run_exists(task.task_id, setup_exists)
            assert run_ids, (run_ids, clf)

    @pytest.mark.sklearn()
    def test_run_with_illegal_flow_id(self):
        # check the case where the user adds an illegal flow id to a
        # non-existing flo
        task = openml.tasks.get_task(115)  # diabetes; crossvalidation
        clf = DecisionTreeClassifier()
        flow = self.extension.model_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        flow.flow_id = -1
        expected_message_regex = (
            r"Flow does not exist on the server, but 'flow.flow_id' is not None."
        )
        with pytest.raises(openml.exceptions.PyOpenMLError, match=expected_message_regex):
            openml.runs.run_flow_on_task(
                task=task,
                flow=flow,
                avoid_duplicate_runs=True,
            )

    @pytest.mark.sklearn()
    def test_run_with_illegal_flow_id_after_load(self):
        # Same as `test_run_with_illegal_flow_id`, but test this error is also
        # caught if the run is stored to and loaded from disk first.
        task = openml.tasks.get_task(115)  # diabetes; crossvalidation
        clf = DecisionTreeClassifier()
        flow = self.extension.model_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        flow.flow_id = -1
        run = openml.runs.run_flow_on_task(
            task=task,
            flow=flow,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        cache_path = os.path.join(
            self.workdir,
            "runs",
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)
        loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)

        expected_message_regex = (
            r"Flow does not exist on the server, but 'flow.flow_id' is not None."
        )
        with pytest.raises(openml.exceptions.PyOpenMLError, match=expected_message_regex):
            loaded_run.publish()
            TestBase._mark_entity_for_removal("run", loaded_run.run_id)
            TestBase.logger.info(f"collected from test_run_functions: {loaded_run.run_id}")

    @pytest.mark.sklearn()
    def test_run_with_illegal_flow_id_1(self):
        # Check the case where the user adds an illegal flow id to an existing
        # flow. Comes to a different value error than the previous test
        task = openml.tasks.get_task(115)  # diabetes; crossvalidation
        clf = DecisionTreeClassifier()
        flow_orig = self.extension.model_to_flow(clf)
        try:
            flow_orig.publish()  # ensures flow exist on server
            TestBase._mark_entity_for_removal("flow", flow_orig.flow_id, flow_orig.name)
            TestBase.logger.info(f"collected from test_run_functions: {flow_orig.flow_id}")
        except openml.exceptions.OpenMLServerException:
            # flow already exists
            pass
        flow_new = self.extension.model_to_flow(clf)

        flow_new.flow_id = -1
        expected_message_regex = "Local flow_id does not match server flow_id: " "'-1' vs '[0-9]+'"
        with pytest.raises(openml.exceptions.PyOpenMLError, match=expected_message_regex):
            openml.runs.run_flow_on_task(
                task=task,
                flow=flow_new,
                avoid_duplicate_runs=True,
            )

    @pytest.mark.sklearn()
    def test_run_with_illegal_flow_id_1_after_load(self):
        # Same as `test_run_with_illegal_flow_id_1`, but test this error is
        # also caught if the run is stored to and loaded from disk first.
        task = openml.tasks.get_task(115)  # diabetes; crossvalidation
        clf = DecisionTreeClassifier()
        flow_orig = self.extension.model_to_flow(clf)
        try:
            flow_orig.publish()  # ensures flow exist on server
            TestBase._mark_entity_for_removal("flow", flow_orig.flow_id, flow_orig.name)
            TestBase.logger.info(f"collected from test_run_functions: {flow_orig.flow_id}")
        except openml.exceptions.OpenMLServerException:
            # flow already exists
            pass
        flow_new = self.extension.model_to_flow(clf)
        flow_new.flow_id = -1

        run = openml.runs.run_flow_on_task(
            task=task,
            flow=flow_new,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        cache_path = os.path.join(
            self.workdir,
            "runs",
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)
        loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)

        expected_message_regex = "Local flow_id does not match server flow_id: " "'-1' vs '[0-9]+'"
        self.assertRaisesRegex(
            openml.exceptions.PyOpenMLError,
            expected_message_regex,
            loaded_run.publish,
        )

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="OneHotEncoder cannot handle mixed type DataFrame as input",
    )
    def test__run_task_get_arffcontent(self):
        task = openml.tasks.get_task(7)  # kr-vs-kp; crossvalidation
        num_instances = 3196
        num_folds = 10
        num_repeats = 1

        clf = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            SGDClassifier(loss="log", random_state=1),
        )
        res = openml.runs.functions._run_task_get_arffcontent(
            extension=self.extension,
            model=clf,
            task=task,
            add_local_measures=True,
            dataset_format="dataframe",
        )
        arff_datacontent, trace, fold_evaluations, _ = res
        # predictions
        assert isinstance(arff_datacontent, list)
        # trace. SGD does not produce any
        assert isinstance(trace, type(None))

        task_type = TaskType.SUPERVISED_CLASSIFICATION
        self._check_fold_timing_evaluations(
            fold_evaluations=fold_evaluations,
            num_repeats=num_repeats,
            num_folds=num_folds,
            task_type=task_type,
        )

        # 10 times 10 fold CV of 150 samples
        assert len(arff_datacontent) == num_instances * num_repeats
        for arff_line in arff_datacontent:
            # check number columns
            assert len(arff_line) == 8
            # check repeat
            assert arff_line[0] >= 0
            assert arff_line[0] <= num_repeats - 1
            # check fold
            assert arff_line[1] >= 0
            assert arff_line[1] <= num_folds - 1
            # check row id
            assert arff_line[2] >= 0
            assert arff_line[2] <= num_instances - 1
            # check prediction and ground truth columns
            assert arff_line[4] in ["won", "nowin"]
            assert arff_line[5] in ["won", "nowin"]
            # check confidences
            self.assertAlmostEqual(sum(arff_line[6:]), 1.0)

    def test__create_trace_from_arff(self):
        with open(self.static_cache_dir / "misc" / "trace.arff") as arff_file:
            trace_arff = arff.load(arff_file)
        OpenMLRunTrace.trace_from_arff(trace_arff)

    @pytest.mark.production()
    def test_get_run(self):
        # this run is not available on test
        openml.config.server = self.production_server
        run = openml.runs.get_run(473351)
        assert run.dataset_id == 357
        assert run.evaluations["f_measure"] == 0.841225
        for i, value in [
            (0, 0.840918),
            (1, 0.839458),
            (2, 0.839613),
            (3, 0.842571),
            (4, 0.839567),
            (5, 0.840922),
            (6, 0.840985),
            (7, 0.847129),
            (8, 0.84218),
            (9, 0.844014),
        ]:
            assert run.fold_evaluations["f_measure"][0][i] == value
        assert "weka" in run.tags
        assert "weka_3.7.12" in run.tags
        assert run.predictions_url == (
            "https://api.openml.org/data/download/1667125/"
            "weka_generated_predictions4575715871712251329.arff"
        )

    def _check_run(self, run):
        # This tests that the API returns seven entries for each run
        # Check out https://openml.org/api/v1/xml/run/list/flow/1154
        # They are run_id, task_id, task_type_id, setup_id, flow_id, uploader, upload_time
        # error_message and run_details exist, too, but are not used so far. We need to update
        # this check once they are used!
        assert isinstance(run, dict)
        assert len(run) == 8, str(run)

    @pytest.mark.production()
    def test_get_runs_list(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        runs = openml.runs.list_runs(id=[2], show_errors=True, output_format="dataframe")
        assert len(runs) == 1
        for run in runs.to_dict(orient="index").values():
            self._check_run(run)

    def test_list_runs_empty(self):
        runs = openml.runs.list_runs(task=[0], output_format="dataframe")
        assert runs.empty

    def test_list_runs_output_format(self):
        runs = openml.runs.list_runs(size=1000, output_format="dataframe")
        assert isinstance(runs, pd.DataFrame)

    @pytest.mark.production()
    def test_get_runs_list_by_task(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        task_ids = [20]
        runs = openml.runs.list_runs(task=task_ids, output_format="dataframe")
        assert len(runs) >= 590
        for run in runs.to_dict(orient="index").values():
            assert run["task_id"] in task_ids
            self._check_run(run)
        num_runs = len(runs)

        task_ids.append(21)
        runs = openml.runs.list_runs(task=task_ids, output_format="dataframe")
        assert len(runs) >= num_runs + 1
        for run in runs.to_dict(orient="index").values():
            assert run["task_id"] in task_ids
            self._check_run(run)

    @pytest.mark.production()
    def test_get_runs_list_by_uploader(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        # 29 is Dominik Kirchhoff
        uploader_ids = [29]

        runs = openml.runs.list_runs(uploader=uploader_ids, output_format="dataframe")
        assert len(runs) >= 2
        for run in runs.to_dict(orient="index").values():
            assert run["uploader"] in uploader_ids
            self._check_run(run)
        num_runs = len(runs)

        uploader_ids.append(274)

        runs = openml.runs.list_runs(uploader=uploader_ids, output_format="dataframe")
        assert len(runs) >= num_runs + 1
        for run in runs.to_dict(orient="index").values():
            assert run["uploader"] in uploader_ids
            self._check_run(run)

    @pytest.mark.production()
    def test_get_runs_list_by_flow(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        flow_ids = [1154]
        runs = openml.runs.list_runs(flow=flow_ids, output_format="dataframe")
        assert len(runs) >= 1
        for run in runs.to_dict(orient="index").values():
            assert run["flow_id"] in flow_ids
            self._check_run(run)
        num_runs = len(runs)

        flow_ids.append(1069)
        runs = openml.runs.list_runs(flow=flow_ids, output_format="dataframe")
        assert len(runs) >= num_runs + 1
        for run in runs.to_dict(orient="index").values():
            assert run["flow_id"] in flow_ids
            self._check_run(run)

    @pytest.mark.production()
    def test_get_runs_pagination(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        uploader_ids = [1]
        size = 10
        max = 100
        for i in range(0, max, size):
            runs = openml.runs.list_runs(
                offset=i,
                size=size,
                uploader=uploader_ids,
                output_format="dataframe",
            )
            assert size >= len(runs)
            for run in runs.to_dict(orient="index").values():
                assert run["uploader"] in uploader_ids

    @pytest.mark.production()
    def test_get_runs_list_by_filters(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        ids = [505212, 6100]
        tasks = [2974, 339]
        uploaders_1 = [1, 2]
        uploaders_2 = [29, 274]
        flows = [74, 1718]

        """
        Since the results are taken by batch size, the function does not
        throw an OpenMLServerError anymore. Instead it throws a
        TimeOutException. For the moment commented out.
        """
        # self.assertRaises(openml.exceptions.OpenMLServerError,
        # openml.runs.list_runs)

        runs = openml.runs.list_runs(id=ids, output_format="dataframe")
        assert len(runs) == 2

        runs = openml.runs.list_runs(task=tasks, output_format="dataframe")
        assert len(runs) >= 2

        runs = openml.runs.list_runs(uploader=uploaders_2, output_format="dataframe")
        assert len(runs) >= 10

        runs = openml.runs.list_runs(flow=flows, output_format="dataframe")
        assert len(runs) >= 100

        runs = openml.runs.list_runs(
            id=ids,
            task=tasks,
            uploader=uploaders_1,
            output_format="dataframe",
        )
        assert len(runs) == 2

    @pytest.mark.production()
    def test_get_runs_list_by_tag(self):
        # TODO: comes from live, no such lists on test
        # Unit test works on production server only
        openml.config.server = self.production_server
        runs = openml.runs.list_runs(tag="curves", output_format="dataframe")
        assert len(runs) >= 1

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    def test_run_on_dataset_with_missing_labels_dataframe(self):
        # Check that _run_task_get_arffcontent works when one of the class
        # labels only declared in the arff file, but is not present in the
        # actual data
        task = openml.tasks.get_task(2)  # anneal; crossvalidation

        from sklearn.compose import ColumnTransformer

        cat_imp = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        cont_imp = make_pipeline(CustomImputer(), StandardScaler())
        ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
        model = Pipeline(
            steps=[("preprocess", ct), ("estimator", sklearn.tree.DecisionTreeClassifier())],
        )  # build a sklearn classifier

        data_content, _, _, _ = _run_task_get_arffcontent(
            model=model,
            task=task,
            extension=self.extension,
            add_local_measures=True,
            dataset_format="dataframe",
        )
        # 2 folds, 5 repeats; keep in mind that this task comes from the test
        # server, the task on the live server is different
        assert len(data_content) == 4490
        for row in data_content:
            # repeat, fold, row_id, 6 confidences, prediction and correct label
            assert len(row) == 12

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="columntransformer introduction in 0.20.0",
    )
    def test_run_on_dataset_with_missing_labels_array(self):
        # Check that _run_task_get_arffcontent works when one of the class
        # labels only declared in the arff file, but is not present in the
        # actual data
        task = openml.tasks.get_task(2)  # anneal; crossvalidation
        # task_id=2 on test server has 38 columns with 6 numeric columns
        cont_idx = [3, 4, 8, 32, 33, 34]
        cat_idx = list(set(np.arange(38)) - set(cont_idx))
        cont = np.array([False] * 38)
        cat = np.array([False] * 38)
        cont[cont_idx] = True
        cat[cat_idx] = True

        from sklearn.compose import ColumnTransformer

        cat_imp = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        cont_imp = make_pipeline(CustomImputer(), StandardScaler())
        ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
        model = Pipeline(
            steps=[("preprocess", ct), ("estimator", sklearn.tree.DecisionTreeClassifier())],
        )  # build a sklearn classifier

        data_content, _, _, _ = _run_task_get_arffcontent(
            model=model,
            task=task,
            extension=self.extension,
            add_local_measures=True,
            dataset_format="array",  # diff test_run_on_dataset_with_missing_labels_dataframe()
        )
        # 2 folds, 5 repeats; keep in mind that this task comes from the test
        # server, the task on the live server is different
        assert len(data_content) == 4490
        for row in data_content:
            # repeat, fold, row_id, 6 confidences, prediction and correct label
            assert len(row) == 12

    def test_get_cached_run(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        openml.runs.functions._get_cached_run(1)

    def test_get_uncached_run(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        with pytest.raises(openml.exceptions.OpenMLCacheException):
            openml.runs.functions._get_cached_run(10)

    @pytest.mark.sklearn()
    def test_run_flow_on_task_downloaded_flow(self):
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=33)
        flow = self.extension.model_to_flow(model)
        flow.publish(raise_error_if_exists=False)
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(f"collected from test_run_functions: {flow.flow_id}")

        downloaded_flow = openml.flows.get_flow(flow.flow_id)
        task = openml.tasks.get_task(self.TEST_SERVER_TASK_SIMPLE["task_id"])
        run = openml.runs.run_flow_on_task(
            flow=downloaded_flow,
            task=task,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], run.run_id))

    @pytest.mark.production()
    def test_format_prediction_non_supervised(self):
        # non-supervised tasks don't exist on the test server
        openml.config.server = self.production_server
        clustering = openml.tasks.get_task(126033, download_data=False)
        ignored_input = [0] * 5
        with pytest.raises(
            NotImplementedError, match=r"Formatting for <class '[\w.]+'> is not supported."
        ):
            format_prediction(clustering, *ignored_input)

    def test_format_prediction_classification_no_probabilities(self):
        classification = openml.tasks.get_task(
            self.TEST_SERVER_TASK_SIMPLE["task_id"],
            download_data=False,
        )
        ignored_input = [0] * 5
        with pytest.raises(ValueError, match="`proba` is required for classification task"):
            format_prediction(classification, *ignored_input, proba=None)

    def test_format_prediction_classification_incomplete_probabilities(self):
        classification = openml.tasks.get_task(
            self.TEST_SERVER_TASK_SIMPLE["task_id"],
            download_data=False,
        )
        ignored_input = [0] * 5
        incomplete_probabilities = {c: 0.2 for c in classification.class_labels[1:]}
        with pytest.raises(ValueError, match="Each class should have a predicted probability"):
            format_prediction(classification, *ignored_input, proba=incomplete_probabilities)

    def test_format_prediction_task_without_classlabels_set(self):
        classification = openml.tasks.get_task(
            self.TEST_SERVER_TASK_SIMPLE["task_id"],
            download_data=False,
        )
        classification.class_labels = None
        ignored_input = [0] * 5
        with pytest.raises(ValueError, match="The classification task must have class labels set"):
            format_prediction(classification, *ignored_input, proba={})

    def test_format_prediction_task_learning_curve_sample_not_set(self):
        learning_curve = openml.tasks.get_task(801, download_data=False)  # diabetes;crossvalidation
        probabilities = {c: 0.2 for c in learning_curve.class_labels}
        ignored_input = [0] * 5
        with pytest.raises(ValueError, match="`sample` can not be none for LearningCurveTask"):
            format_prediction(learning_curve, *ignored_input, sample=None, proba=probabilities)

    def test_format_prediction_task_regression(self):
        task_meta_data = self.TEST_SERVER_TASK_REGRESSION["task_meta_data"]
        _task_id = check_task_existence(**task_meta_data)
        if _task_id is not None:
            task_id = _task_id
        else:
            new_task = openml.tasks.create_task(**task_meta_data)
            # publishes the new task
            try:
                new_task = new_task.publish()
                task_id = new_task.task_id
            except OpenMLServerException as e:
                if e.code == 614:  # Task already exists
                    # the exception message contains the task_id that was matched in the format
                    # 'Task already exists. - matched id(s): [xxxx]'
                    task_id = ast.literal_eval(e.message.split("matched id(s):")[-1].strip())[0]
                else:
                    raise Exception(repr(e))
            # mark to remove the uploaded task
            TestBase._mark_entity_for_removal("task", task_id)
            TestBase.logger.info(f"collected from test_run_functions: {task_id}")

        regression = openml.tasks.get_task(task_id, download_data=False)
        ignored_input = [0] * 5
        res = format_prediction(regression, *ignored_input)
        self.assertListEqual(res, [0] * 5)

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.21",
        reason="couldn't perform local tests successfully w/o bloating RAM",
    )
    @mock.patch("openml.extensions.sklearn.SklearnExtension._prevent_optimize_n_jobs")
    def test__run_task_get_arffcontent_2(self, parallel_mock):
        """Tests if a run executed in parallel is collated correctly."""
        task = openml.tasks.get_task(7)  # Supervised Classification on kr-vs-kp
        x, y = task.get_X_and_y(dataset_format="dataframe")
        num_instances = x.shape[0]
        line_length = 6 + len(task.class_labels)
        clf = SGDClassifier(loss="log", random_state=1)
        n_jobs = 2
        backend = "loky" if LooseVersion(joblib.__version__) > "0.11" else "multiprocessing"
        with parallel_backend(backend, n_jobs=n_jobs):
            res = openml.runs.functions._run_task_get_arffcontent(
                extension=self.extension,
                model=clf,
                task=task,
                add_local_measures=True,
                dataset_format="array",  # "dataframe" would require handling of categoricals
                n_jobs=n_jobs,
            )
        # This unit test will fail if joblib is unable to distribute successfully since the
        # function _run_model_on_fold is being mocked out. However, for a new spawned worker, it
        # is not and the mock call_count should remain 0 while the subsequent check of actual
        # results should also hold, only on successful distribution of tasks to workers.
        # The _prevent_optimize_n_jobs() is a function executed within the _run_model_on_fold()
        # block and mocking this function doesn't affect rest of the pipeline, but is adequately
        # indicative if _run_model_on_fold() is being called or not.
        assert parallel_mock.call_count == 0
        assert isinstance(res[0], list)
        assert len(res[0]) == num_instances
        assert len(res[0][0]) == line_length
        assert len(res[2]) == 7
        assert len(res[3]) == 7
        expected_scores = [
            0.965625,
            0.94375,
            0.946875,
            0.953125,
            0.96875,
            0.965625,
            0.9435736677115988,
            0.9467084639498433,
            0.9749216300940439,
            0.9655172413793104,
        ]
        scores = [v for k, v in res[2]["predictive_accuracy"][0].items()]
        np.testing.assert_array_almost_equal(
            scores,
            expected_scores,
            decimal=2 if os.name == "nt" else 7,
        )

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.21",
        reason="couldn't perform local tests successfully w/o bloating RAM",
    )
    @mock.patch("openml.extensions.sklearn.SklearnExtension._prevent_optimize_n_jobs")
    def test_joblib_backends(self, parallel_mock):
        """Tests evaluation of a run using various joblib backends and n_jobs."""
        task = openml.tasks.get_task(7)  # Supervised Classification on kr-vs-kp
        x, y = task.get_X_and_y(dataset_format="dataframe")
        num_instances = x.shape[0]
        line_length = 6 + len(task.class_labels)

        backend_choice = "loky" if LooseVersion(joblib.__version__) > "0.11" else "multiprocessing"
        for n_jobs, backend, call_count in [
            (1, backend_choice, 10),
            (2, backend_choice, 10),
            (-1, backend_choice, 10),
            (1, "threading", 20),
            (-1, "threading", 30),
            (1, "sequential", 40),
        ]:
            clf = sklearn.model_selection.RandomizedSearchCV(
                estimator=sklearn.ensemble.RandomForestClassifier(n_estimators=5),
                param_distributions={
                    "max_depth": [3, None],
                    "max_features": [1, 2, 3, 4],
                    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                },
                random_state=1,
                cv=sklearn.model_selection.StratifiedKFold(
                    n_splits=2,
                    shuffle=True,
                    random_state=1,
                ),
                n_iter=5,
                n_jobs=n_jobs,
            )
            with parallel_backend(backend, n_jobs=n_jobs):
                res = openml.runs.functions._run_task_get_arffcontent(
                    extension=self.extension,
                    model=clf,
                    task=task,
                    add_local_measures=True,
                    dataset_format="array",  # "dataframe" would require handling of categoricals
                    n_jobs=n_jobs,
                )
            assert type(res[0]) == list
            assert len(res[0]) == num_instances
            assert len(res[0][0]) == line_length
            # usercpu_time_millis_* not recorded when n_jobs > 1
            # *_time_millis_* not recorded when n_jobs = -1
            assert len(res[2]["predictive_accuracy"][0]) == 10
            assert len(res[3]["predictive_accuracy"][0]) == 10
            assert parallel_mock.call_count == call_count

    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.20",
        reason="SimpleImputer doesn't handle mixed type DataFrame as input",
    )
    def test_delete_run(self):
        rs = 1
        clf = sklearn.pipeline.Pipeline(
            steps=[("imputer", SimpleImputer()), ("estimator", DecisionTreeClassifier())],
        )
        task = openml.tasks.get_task(32)  # diabetes; crossvalidation

        run = openml.runs.run_model_on_task(model=clf, task=task, seed=rs)
        run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info(f"collected from test_run_functions: {run.run_id}")

        _run_id = run.run_id
        assert delete_run(_run_id)


@mock.patch.object(requests.Session, "delete")
def test_delete_run_not_owned(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "runs" / "run_delete_not_owned.xml"
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLNotAuthorizedError,
        match="The run can not be deleted because it was not uploaded by you.",
    ):
        openml.runs.delete_run(40_000)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/run/40000",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_run_success(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "runs" / "run_delete_successful.xml"
    mock_delete.return_value = create_request_response(
        status_code=200,
        content_filepath=content_file,
    )

    success = openml.runs.delete_run(10591880)
    assert success

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/run/10591880",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)


@mock.patch.object(requests.Session, "delete")
def test_delete_unknown_run(mock_delete, test_files_directory, test_api_key):
    openml.config.start_using_configuration_for_example()
    content_file = test_files_directory / "mock_responses" / "runs" / "run_delete_not_exist.xml"
    mock_delete.return_value = create_request_response(
        status_code=412,
        content_filepath=content_file,
    )

    with pytest.raises(
        OpenMLServerException,
        match="Run does not exist",
    ):
        openml.runs.delete_run(9_999_999)

    expected_call_args = [
        ("https://test.openml.org/api/v1/xml/run/9999999",),
        {"params": {"api_key": test_api_key}},
    ]
    assert expected_call_args == list(mock_delete.call_args)
