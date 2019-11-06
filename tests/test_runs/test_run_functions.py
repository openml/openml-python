# License: BSD 3-Clause

import arff
from distutils.version import LooseVersion
import os
import random
import time
import sys
import unittest.mock

import numpy as np
import pytest

import openml
import openml.exceptions
import openml._api_calls
import sklearn
import unittest
import warnings
import pandas as pd

import openml.extensions.sklearn
from openml.testing import TestBase, SimpleImputer
from openml.runs.functions import (
    _run_task_get_arffcontent,
    run_exists,
)
from openml.runs.trace import OpenMLRunTrace
from openml.tasks import TaskTypeEnum

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection._search import BaseSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, SGDClassifier, \
    LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, \
    StratifiedKFold
from sklearn.pipeline import Pipeline


class TestRun(TestBase):
    _multiprocess_can_split_ = True
    # diabetis dataset, 768 observations, 0 missing vals, 33% holdout set
    # (253 test obs), no nominal attributes, all numeric attributes
    TEST_SERVER_TASK_SIMPLE = (119, 0, 253, list(), list(range(8)))
    TEST_SERVER_TASK_REGRESSION = (738, 0, 718, list(), list(range(8)))
    # credit-a dataset, 690 observations, 67 missing vals, 33% holdout set
    # (227 test obs)
    TEST_SERVER_TASK_MISSING_VALS = (96, 67, 227,
                                     [0, 3, 4, 5, 6, 8, 9, 11, 12],
                                     [1, 2, 7, 10, 13, 14])

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
            if len(run.evaluations) > 0:
                return
            else:
                time.sleep(3)
        raise RuntimeError('Could not find any evaluations! Please check whether run {} was '
                           'evaluated correctly on the server'.format(run_id))

    def _compare_predictions(self, predictions, predictions_prime):
        self.assertEqual(np.array(predictions_prime['data']).shape,
                         np.array(predictions['data']).shape)

        # The original search model does not submit confidence
        # bounds, so we can not compare the arff line
        compare_slice = [0, 1, 2, -1, -2]
        for idx in range(len(predictions['data'])):
            # depends on the assumption "predictions are in same order"
            # that does not necessarily hold.
            # But with the current code base, it holds.
            for col_idx in compare_slice:
                val_1 = predictions['data'][idx][col_idx]
                val_2 = predictions_prime['data'][idx][col_idx]
                if type(val_1) == float or type(val_2) == float:
                    self.assertAlmostEqual(
                        float(val_1),
                        float(val_2),
                        places=6,
                    )
                else:
                    self.assertEqual(val_1, val_2)

        return True

    def _rerun_model_and_compare_predictions(self, run_id, model_prime, seed):
        run = openml.runs.get_run(run_id)
        task = openml.tasks.get_task(run.task_id)

        # TODO: assert holdout task

        # downloads the predictions of the old task
        file_id = run.output_files['predictions']
        predictions_url = openml._api_calls._file_id_to_url(file_id)
        response = openml._api_calls._read_url(predictions_url,
                                               request_method='get')
        predictions = arff.loads(response)
        run_prime = openml.runs.run_model_on_task(
            model=model_prime,
            task=task,
            avoid_duplicate_runs=False,
            seed=seed,
        )
        predictions_prime = run_prime._generate_arff_dict()

        self._compare_predictions(predictions, predictions_prime)

    def _perform_run(self, task_id, num_instances, n_missing_vals, clf,
                     flow_expected_rsv=None, seed=1, check_setup=True,
                     sentinel=None):
        """
        Runs a classifier on a task, and performs some basic checks.
        Also uploads the run.

        Parameters:
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

        Returns:
        --------
        run: OpenMLRun
            The performed run (with run id)
        """
        classes_without_random_state = \
            ['sklearn.model_selection._search.GridSearchCV',
             'sklearn.pipeline.Pipeline',
             'sklearn.linear_model.base.LinearRegression',
             ]

        def _remove_random_state(flow):
            if 'random_state' in flow.parameters:
                del flow.parameters['random_state']
            for component in flow.components.values():
                _remove_random_state(component)

        flow = self.extension.model_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, sentinel)
        if not openml.flows.flow_exists(flow.name, flow.external_version):
            flow.publish()
            TestBase._mark_entity_for_removal('flow', (flow.flow_id, flow.name))
            TestBase.logger.info("collected from test_run_functions: {}".format(flow.flow_id))

        task = openml.tasks.get_task(task_id)

        X, y = task.get_X_and_y()
        self.assertEqual(np.count_nonzero(np.isnan(X)), n_missing_vals)
        run = openml.runs.run_flow_on_task(
            flow=flow,
            task=task,
            seed=seed,
            avoid_duplicate_runs=openml.config.avoid_duplicate_runs,
        )
        run_ = run.publish()
        TestBase._mark_entity_for_removal('run', run.run_id)
        TestBase.logger.info("collected from test_run_functions: {}".format(run.run_id))
        self.assertEqual(run_, run)
        self.assertIsInstance(run.dataset_id, int)

        # This is only a smoke check right now
        # TODO add a few asserts here
        run._to_xml()
        if run.trace is not None:
            # This is only a smoke check right now
            # TODO add a few asserts here
            run.trace.trace_to_arff()

        # check arff output
        self.assertEqual(len(run.data_content), num_instances)

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
                error_msg = 'Flow class %s (id=%d) does not have a random ' \
                            'state parameter' % (flow.class_name, flow.flow_id)
                self.assertIn('random_state', flow.parameters, error_msg)
                # If the flow is initialized from a model without a random
                # state, the flow is on the server without any random state
                self.assertEqual(flow.parameters['random_state'], 'null')
                # As soon as a flow is run, a random state is set in the model.
                # If a flow is re-instantiated
                self.assertEqual(flow_local.parameters['random_state'],
                                 flow_expected_rsv)
                self.assertEqual(flow_server.parameters['random_state'],
                                 flow_expected_rsv)
            _remove_random_state(flow_local)
            _remove_random_state(flow_server)
            openml.flows.assert_flows_equal(flow_local, flow_server)

            # and test the initialize setup from run function
            clf_server2 = openml.runs.initialize_model_from_run(
                run_id=run_server.run_id,
            )
            flow_server2 = self.extension.model_to_flow(clf_server2)
            if flow.class_name not in classes_without_random_state:
                self.assertEqual(flow_server2.parameters['random_state'],
                                 flow_expected_rsv)

            _remove_random_state(flow_server2)
            openml.flows.assert_flows_equal(flow_local, flow_server2)

            # self.assertEqual(clf.get_params(), clf_prime.get_params())
            # self.assertEqual(clf, clf_prime)

        downloaded = openml.runs.get_run(run_.run_id)
        assert ('openml-python' in downloaded.tags)

        # TODO make sure that these attributes are instantiated when
        # downloading a run? Or make sure that the trace object is created when
        # running a flow on a task (and not only the arff object is created,
        # so that the two objects can actually be compared):
        # downloaded_run_trace = downloaded._generate_trace_arff_dict()
        # self.assertEqual(run_trace, downloaded_run_trace)
        return run

    def _check_sample_evaluations(self, sample_evaluations, num_repeats,
                                  num_folds, num_samples,
                                  max_time_allowed=60000):
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
            'usercpu_time_millis_testing': (0, max_time_allowed),
            'usercpu_time_millis_training': (0, max_time_allowed),
            'usercpu_time_millis': (0, max_time_allowed),
            'wall_clock_time_millis_training': (0, max_time_allowed),
            'wall_clock_time_millis_testing': (0, max_time_allowed),
            'wall_clock_time_millis': (0, max_time_allowed),
            'predictive_accuracy': (0, 1)
        }

        self.assertIsInstance(sample_evaluations, dict)
        if sys.version_info[:2] >= (3, 3):
            # this only holds if we are allowed to record time (otherwise some
            # are missing)
            self.assertEqual(set(sample_evaluations.keys()),
                             set(check_measures.keys()))

        for measure in check_measures.keys():
            if measure in sample_evaluations:
                num_rep_entrees = len(sample_evaluations[measure])
                self.assertEqual(num_rep_entrees, num_repeats)
                for rep in range(num_rep_entrees):
                    num_fold_entrees = len(sample_evaluations[measure][rep])
                    self.assertEqual(num_fold_entrees, num_folds)
                    for fold in range(num_fold_entrees):
                        num_sample_entrees = len(
                            sample_evaluations[measure][rep][fold])
                        self.assertEqual(num_sample_entrees, num_samples)
                        for sample in range(num_sample_entrees):
                            evaluation = sample_evaluations[measure][rep][
                                fold][sample]
                            self.assertIsInstance(evaluation, float)
                            if not os.environ.get('CI_WINDOWS'):
                                # Either Appveyor is much faster than Travis
                                # and/or measurements are not as accurate.
                                # Either way, windows seems to get an eval-time
                                # of 0 sometimes.
                                self.assertGreater(evaluation, 0)
                            self.assertLess(evaluation, max_time_allowed)

    def test_run_regression_on_classif_task(self):
        task_id = 115

        clf = LinearRegression()
        task = openml.tasks.get_task(task_id)
        with self.assertRaises(AttributeError):
            openml.runs.run_model_on_task(
                model=clf,
                task=task,
                avoid_duplicate_runs=False,
            )

    def test_check_erronous_sklearn_flow_fails(self):
        task_id = 115
        task = openml.tasks.get_task(task_id)

        # Invalid parameter values
        clf = LogisticRegression(C='abc', solver='lbfgs')
        with self.assertRaisesRegex(
            ValueError,
            r"Penalty term must be positive; got \(C=u?'abc'\)",  # u? for 2.7/3.4-6 compability
        ):
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

    def _run_and_upload(self, clf, task_id, n_missing_vals, n_test_obs,
                        flow_expected_rsv, num_folds=1, num_iterations=5,
                        seed=1, metric=sklearn.metrics.accuracy_score,
                        metric_name='predictive_accuracy',
                        task_type=TaskTypeEnum.SUPERVISED_CLASSIFICATION,
                        sentinel=None):
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
                raise TypeError('Param Grid should be of type list '
                                '(GridSearch only) or dict')

        run = self._perform_run(task_id, n_test_obs, n_missing_vals, clf,
                                flow_expected_rsv=flow_expected_rsv, seed=seed,
                                sentinel=sentinel)

        # obtain scores using get_metric_score:
        scores = run.get_metric_fn(metric)
        # compare with the scores in user defined measures
        scores_provided = []
        for rep in run.fold_evaluations[metric_name].keys():
            for fold in run.fold_evaluations[metric_name][rep].keys():
                scores_provided.append(
                    run.fold_evaluations[metric_name][rep][fold])
        self.assertEqual(sum(scores_provided), sum(scores))

        if isinstance(clf, BaseSearchCV):
            trace_content = run.trace.trace_to_arff()['data']
            if isinstance(clf, GridSearchCV):
                grid_iterations = determine_grid_size(clf.param_grid)
                self.assertEqual(len(trace_content),
                                 grid_iterations * num_folds)
            else:
                self.assertEqual(len(trace_content),
                                 num_iterations * num_folds)

            # downloads the best model based on the optimization trace
            # suboptimal (slow), and not guaranteed to work if evaluation
            # engine is behind.
            # TODO: mock this? We have the arff already on the server
            self._wait_for_processed_run(run.run_id, 400)
            try:
                model_prime = openml.runs.initialize_model_from_trace(
                    run_id=run.run_id,
                    repeat=0,
                    fold=0,
                )
            except openml.exceptions.OpenMLServerException as e:
                e.message = "%s; run_id %d" % (e.message, run.run_id)
                raise e

            self._rerun_model_and_compare_predictions(run.run_id, model_prime,
                                                      seed)
        else:
            run_downloaded = openml.runs.get_run(run.run_id)
            sid = run_downloaded.setup_id
            model_prime = openml.setups.initialize_model(sid)
            self._rerun_model_and_compare_predictions(run.run_id,
                                                      model_prime, seed)

        # todo: check if runtime is present
        self._check_fold_timing_evaluations(run.fold_evaluations, 1, num_folds,
                                            task_type=task_type)
        return run

    def _run_and_upload_classification(self, clf, task_id, n_missing_vals,
                                       n_test_obs, flow_expected_rsv,
                                       sentinel=None):
        num_folds = 1  # because of holdout
        num_iterations = 5  # for base search algorithms
        metric = sklearn.metrics.accuracy_score  # metric class
        metric_name = 'predictive_accuracy'  # openml metric name
        task_type = TaskTypeEnum.SUPERVISED_CLASSIFICATION  # task type

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

    def _run_and_upload_regression(self, clf, task_id, n_missing_vals,
                                   n_test_obs, flow_expected_rsv,
                                   sentinel=None):
        num_folds = 1  # because of holdout
        num_iterations = 5  # for base search algorithms
        metric = sklearn.metrics.mean_absolute_error  # metric class
        metric_name = 'mean_absolute_error'  # openml metric name
        task_type = TaskTypeEnum.SUPERVISED_REGRESSION  # task type

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

    def test_run_and_upload_logistic_regression(self):
        lr = LogisticRegression(solver='lbfgs')
        task_id = self.TEST_SERVER_TASK_SIMPLE[0]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE[1]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE[2]
        self._run_and_upload_classification(lr, task_id, n_missing_vals,
                                            n_test_obs, '62501')

    def test_run_and_upload_linear_regression(self):
        lr = LinearRegression()
        task_id = self.TEST_SERVER_TASK_REGRESSION[0]
        n_missing_vals = self.TEST_SERVER_TASK_REGRESSION[1]
        n_test_obs = self.TEST_SERVER_TASK_REGRESSION[2]
        self._run_and_upload_regression(lr, task_id, n_missing_vals,
                                        n_test_obs, '62501')

    def test_run_and_upload_pipeline_dummy_pipeline(self):

        pipeline1 = Pipeline(steps=[('scaler',
                                     StandardScaler(with_mean=False)),
                                    ('dummy',
                                     DummyClassifier(strategy='prior'))])
        task_id = self.TEST_SERVER_TASK_SIMPLE[0]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE[1]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE[2]
        self._run_and_upload_classification(pipeline1, task_id, n_missing_vals,
                                            n_test_obs, '62501')

    @unittest.skipIf(LooseVersion(sklearn.__version__) < "0.20",
                     reason="columntransformer introduction in 0.20.0")
    def test_run_and_upload_column_transformer_pipeline(self):
        import sklearn.compose
        import sklearn.impute

        def get_ct_cf(nominal_indices, numeric_indices):
            inner = sklearn.compose.ColumnTransformer(
                transformers=[
                    ('numeric', sklearn.preprocessing.StandardScaler(),
                     nominal_indices),
                    ('nominal', sklearn.preprocessing.OneHotEncoder(
                        handle_unknown='ignore'), numeric_indices)],
                remainder='passthrough')
            return sklearn.pipeline.Pipeline(
                steps=[
                    ('imputer', sklearn.impute.SimpleImputer(
                        strategy='constant', fill_value=-1)),
                    ('transformer', inner),
                    ('classifier', sklearn.tree.DecisionTreeClassifier())
                ]
            )

        sentinel = self._get_sentinel()
        self._run_and_upload_classification(
            get_ct_cf(self.TEST_SERVER_TASK_SIMPLE[3],
                      self.TEST_SERVER_TASK_SIMPLE[4]),
            self.TEST_SERVER_TASK_SIMPLE[0], self.TEST_SERVER_TASK_SIMPLE[1],
            self.TEST_SERVER_TASK_SIMPLE[2], '62501', sentinel=sentinel)
        # Due to #602, it is important to test this model on two tasks
        # with different column specifications
        self._run_and_upload_classification(
            get_ct_cf(self.TEST_SERVER_TASK_MISSING_VALS[3],
                      self.TEST_SERVER_TASK_MISSING_VALS[4]),
            self.TEST_SERVER_TASK_MISSING_VALS[0],
            self.TEST_SERVER_TASK_MISSING_VALS[1],
            self.TEST_SERVER_TASK_MISSING_VALS[2],
            '62501', sentinel=sentinel)

    def test_run_and_upload_decision_tree_pipeline(self):
        pipeline2 = Pipeline(steps=[('Imputer', SimpleImputer(strategy='median')),
                                    ('VarianceThreshold', VarianceThreshold()),
                                    ('Estimator', RandomizedSearchCV(
                                        DecisionTreeClassifier(),
                                        {'min_samples_split':
                                         [2 ** x for x in range(1, 8)],
                                         'min_samples_leaf':
                                         [2 ** x for x in range(0, 7)]},
                                        cv=3, n_iter=10))])
        task_id = self.TEST_SERVER_TASK_MISSING_VALS[0]
        n_missing_vals = self.TEST_SERVER_TASK_MISSING_VALS[1]
        n_test_obs = self.TEST_SERVER_TASK_MISSING_VALS[2]
        self._run_and_upload_classification(pipeline2, task_id, n_missing_vals,
                                            n_test_obs, '62501')

    def test_run_and_upload_gridsearch(self):
        gridsearch = GridSearchCV(BaggingClassifier(base_estimator=SVC()),
                                  {"base_estimator__C": [0.01, 0.1, 10],
                                   "base_estimator__gamma": [0.01, 0.1, 10]})
        task_id = self.TEST_SERVER_TASK_SIMPLE[0]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE[1]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE[2]
        run = self._run_and_upload_classification(
            clf=gridsearch,
            task_id=task_id,
            n_missing_vals=n_missing_vals,
            n_test_obs=n_test_obs,
            flow_expected_rsv='62501',
        )
        self.assertEqual(len(run.trace.trace_iterations), 9)

    def test_run_and_upload_randomsearch(self):
        randomsearch = RandomizedSearchCV(
            RandomForestClassifier(n_estimators=5),
            {"max_depth": [3, None],
             "max_features": [1, 2, 3, 4],
             "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
             "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "bootstrap": [True, False],
             "criterion": ["gini", "entropy"]},
            cv=StratifiedKFold(n_splits=2, shuffle=True),
            n_iter=5)
        # The random states for the RandomizedSearchCV is set after the
        # random state of the RandomForestClassifier is set, therefore,
        # it has a different value than the other examples before
        task_id = self.TEST_SERVER_TASK_SIMPLE[0]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE[1]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE[2]
        run = self._run_and_upload_classification(
            clf=randomsearch,
            task_id=task_id,
            n_missing_vals=n_missing_vals,
            n_test_obs=n_test_obs,
            flow_expected_rsv='12172',
        )
        self.assertEqual(len(run.trace.trace_iterations), 5)

    def test_run_and_upload_maskedarrays(self):
        # This testcase is important for 2 reasons:
        # 1) it verifies the correct handling of masked arrays (not all
        # parameters are active)
        # 2) it verifies the correct handling of a 2-layered grid search
        gridsearch = GridSearchCV(
            RandomForestClassifier(n_estimators=5),
            [
                {'max_features': [2, 4]},
                {'min_samples_leaf': [1, 10]}
            ],
            cv=StratifiedKFold(n_splits=2, shuffle=True)
        )
        # The random states for the GridSearchCV is set after the
        # random state of the RandomForestClassifier is set, therefore,
        # it has a different value than the other examples before
        task_id = self.TEST_SERVER_TASK_SIMPLE[0]
        n_missing_vals = self.TEST_SERVER_TASK_SIMPLE[1]
        n_test_obs = self.TEST_SERVER_TASK_SIMPLE[2]
        self._run_and_upload_classification(gridsearch, task_id,
                                            n_missing_vals, n_test_obs,
                                            '12172')

    ##########################################################################

    def test_learning_curve_task_1(self):
        task_id = 801  # diabates dataset
        num_test_instances = 6144  # for learning curve
        num_missing_vals = 0
        num_repeats = 1
        num_folds = 10
        num_samples = 8

        pipeline1 = Pipeline(steps=[('scaler',
                                     StandardScaler(with_mean=False)),
                                    ('dummy',
                                     DummyClassifier(strategy='prior'))])
        run = self._perform_run(task_id, num_test_instances, num_missing_vals,
                                pipeline1, flow_expected_rsv='62501')
        self._check_sample_evaluations(run.sample_evaluations, num_repeats,
                                       num_folds, num_samples)

    def test_learning_curve_task_2(self):
        task_id = 801  # diabates dataset
        num_test_instances = 6144  # for learning curve
        num_missing_vals = 0
        num_repeats = 1
        num_folds = 10
        num_samples = 8

        pipeline2 = Pipeline(steps=[('Imputer', SimpleImputer(strategy='median')),
                                    ('VarianceThreshold', VarianceThreshold()),
                                    ('Estimator', RandomizedSearchCV(
                                        DecisionTreeClassifier(),
                                        {'min_samples_split':
                                         [2 ** x for x in range(1, 8)],
                                         'min_samples_leaf':
                                         [2 ** x for x in range(0, 7)]},
                                        cv=3, n_iter=10))])
        run = self._perform_run(task_id, num_test_instances, num_missing_vals,
                                pipeline2, flow_expected_rsv='62501')
        self._check_sample_evaluations(run.sample_evaluations, num_repeats,
                                       num_folds, num_samples)

    def test_initialize_cv_from_run(self):
        randomsearch = RandomizedSearchCV(
            RandomForestClassifier(n_estimators=5),
            {"max_depth": [3, None],
             "max_features": [1, 2, 3, 4],
             "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
             "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "bootstrap": [True, False],
             "criterion": ["gini", "entropy"]},
            cv=StratifiedKFold(n_splits=2, shuffle=True),
            n_iter=2)

        task = openml.tasks.get_task(11)
        run = openml.runs.run_model_on_task(
            model=randomsearch,
            task=task,
            avoid_duplicate_runs=False,
            seed=1,
        )
        run_ = run.publish()
        TestBase._mark_entity_for_removal('run', run.run_id)
        TestBase.logger.info("collected from test_run_functions: {}".format(run.run_id))
        run = openml.runs.get_run(run_.run_id)

        modelR = openml.runs.initialize_model_from_run(run_id=run.run_id)
        modelS = openml.setups.initialize_model(setup_id=run.setup_id)

        self.assertEqual(modelS.cv.random_state, 62501)
        self.assertEqual(modelR.cv.random_state, 62501)

    def _test_local_evaluations(self, run):

        # compare with the scores in user defined measures
        accuracy_scores_provided = []
        for rep in run.fold_evaluations['predictive_accuracy'].keys():
            for fold in run.fold_evaluations['predictive_accuracy'][rep].\
                    keys():
                accuracy_scores_provided.append(
                    run.fold_evaluations['predictive_accuracy'][rep][fold])
        accuracy_scores = run.get_metric_fn(sklearn.metrics.accuracy_score)
        np.testing.assert_array_almost_equal(accuracy_scores_provided,
                                             accuracy_scores)

        # also check if we can obtain some other scores:
        tests = [(sklearn.metrics.cohen_kappa_score, {'weights': None}),
                 (sklearn.metrics.roc_auc_score, {}),
                 (sklearn.metrics.average_precision_score, {}),
                 (sklearn.metrics.jaccard_similarity_score, {}),
                 (sklearn.metrics.precision_score, {'average': 'macro'}),
                 (sklearn.metrics.brier_score_loss, {})]
        for test_idx, test in enumerate(tests):
            alt_scores = run.get_metric_fn(
                sklearn_fn=test[0],
                kwargs=test[1],
            )
            self.assertEqual(len(alt_scores), 10)
            for idx in range(len(alt_scores)):
                self.assertGreaterEqual(alt_scores[idx], 0)
                self.assertLessEqual(alt_scores[idx], 1)

    def test_local_run_swapped_parameter_order_model(self):

        # construct sci-kit learn classifier
        clf = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                              ('estimator', RandomForestClassifier())])

        # download task
        task = openml.tasks.get_task(7)

        # invoke OpenML run
        run = openml.runs.run_model_on_task(
            task, clf,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        self._test_local_evaluations(run)

    def test_local_run_swapped_parameter_order_flow(self):

        # construct sci-kit learn classifier
        clf = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                              ('estimator', RandomForestClassifier())])

        flow = self.extension.model_to_flow(clf)
        # download task
        task = openml.tasks.get_task(7)

        # invoke OpenML run
        run = openml.runs.run_flow_on_task(
            task, flow,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        self._test_local_evaluations(run)

    def test_local_run_metric_score(self):

        # construct sci-kit learn classifier
        clf = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                              ('estimator', RandomForestClassifier())])

        # download task
        task = openml.tasks.get_task(7)

        # invoke OpenML run
        run = openml.runs.run_model_on_task(
            model=clf,
            task=task,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        self._test_local_evaluations(run)

    def test_online_run_metric_score(self):
        openml.config.server = self.production_server

        # important to use binary classification task,
        # due to assertions
        run = openml.runs.get_run(9864498)

        self._test_local_evaluations(run)

    def test_initialize_model_from_run(self):
        clf = sklearn.pipeline.Pipeline(steps=[
            ('Imputer', SimpleImputer(strategy='median')),
            ('VarianceThreshold', VarianceThreshold(threshold=0.05)),
            ('Estimator', GaussianNB())])
        task = openml.tasks.get_task(11)
        run = openml.runs.run_model_on_task(
            model=clf,
            task=task,
            avoid_duplicate_runs=False,
        )
        run_ = run.publish()
        TestBase._mark_entity_for_removal('run', run_.run_id)
        TestBase.logger.info("collected from test_run_functions: {}".format(run_.run_id))
        run = openml.runs.get_run(run_.run_id)

        modelR = openml.runs.initialize_model_from_run(run_id=run.run_id)
        modelS = openml.setups.initialize_model(setup_id=run.setup_id)

        flowR = self.extension.model_to_flow(modelR)
        flowS = self.extension.model_to_flow(modelS)
        flowL = self.extension.model_to_flow(clf)
        openml.flows.assert_flows_equal(flowR, flowL)
        openml.flows.assert_flows_equal(flowS, flowL)

        self.assertEqual(flowS.components['Imputer'].
                         parameters['strategy'], '"median"')
        self.assertEqual(flowS.components['VarianceThreshold'].
                         parameters['threshold'], '0.05')

    @pytest.mark.flaky()
    def test_get_run_trace(self):
        # get_run_trace is already tested implicitly in test_run_and_publish
        # this test is a bit additional.
        num_iterations = 10
        num_folds = 1
        task_id = 119

        task = openml.tasks.get_task(task_id)

        # IMPORTANT! Do not sentinel this flow. is faster if we don't wait
        # on openml server
        clf = RandomizedSearchCV(RandomForestClassifier(random_state=42,
                                                        n_estimators=5),

                                 {"max_depth": [3, None],
                                  "max_features": [1, 2, 3, 4],
                                  "bootstrap": [True, False],
                                  "criterion": ["gini", "entropy"]},
                                 num_iterations, random_state=42, cv=3)

        # [SPEED] make unit test faster by exploiting run information
        # from the past
        try:
            # in case the run did not exists yet
            run = openml.runs.run_model_on_task(
                model=clf,
                task=task,
                avoid_duplicate_runs=True,
            )

            self.assertEqual(
                len(run.trace.trace_iterations),
                num_iterations * num_folds,
            )
            run = run.publish()
            TestBase._mark_entity_for_removal('run', run.run_id)
            TestBase.logger.info("collected from test_run_functions: {}".format(run.run_id))
            self._wait_for_processed_run(run.run_id, 200)
            run_id = run.run_id
        except openml.exceptions.OpenMLRunsExistError as e:
            # The only error we expect, should fail otherwise.
            run_ids = [int(run_id) for run_id in e.run_ids]
            self.assertGreater(len(run_ids), 0)
            run_id = random.choice(list(run_ids))

        # now the actual unit test ...
        run_trace = openml.runs.get_run_trace(run_id)
        self.assertEqual(len(run_trace.trace_iterations), num_iterations * num_folds)

    def test__run_exists(self):
        # would be better to not sentinel these clfs,
        # so we do not have to perform the actual runs
        # and can just check their status on line
        rs = 1
        clfs = [
            sklearn.pipeline.Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy='mean')),
                ('VarianceThreshold', VarianceThreshold(threshold=0.05)),
                ('Estimator', DecisionTreeClassifier(max_depth=4))
            ]),
            sklearn.pipeline.Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy='most_frequent')),
                ('VarianceThreshold', VarianceThreshold(threshold=0.1)),
                ('Estimator', DecisionTreeClassifier(max_depth=4))]
            )
        ]

        task = openml.tasks.get_task(115)

        for clf in clfs:
            try:
                # first populate the server with this run.
                # skip run if it was already performed.
                run = openml.runs.run_model_on_task(
                    model=clf,
                    task=task,
                    seed=rs,
                    avoid_duplicate_runs=True,
                    upload_flow=True
                )
                run.publish()
                TestBase._mark_entity_for_removal('run', run.run_id)
                TestBase.logger.info("collected from test_run_functions: {}".format(run.run_id))
            except openml.exceptions.PyOpenMLError:
                # run already existed. Great.
                pass

            flow = self.extension.model_to_flow(clf)
            flow_exists = openml.flows.flow_exists(flow.name, flow.external_version)
            self.assertGreater(flow_exists, 0)
            # Do NOT use get_flow reinitialization, this potentially sets
            # hyperparameter values wrong. Rather use the local model.
            downloaded_flow = openml.flows.get_flow(flow_exists)
            downloaded_flow.model = clf
            setup_exists = openml.setups.setup_exists(downloaded_flow)
            self.assertGreater(setup_exists, 0)
            run_ids = run_exists(task.task_id, setup_exists)
            self.assertTrue(run_ids, msg=(run_ids, clf))

    def test_run_with_illegal_flow_id(self):
        # check the case where the user adds an illegal flow id to a
        # non-existing flow
        task = openml.tasks.get_task(115)
        clf = DecisionTreeClassifier()
        flow = self.extension.model_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        flow.flow_id = -1
        expected_message_regex = ("Flow does not exist on the server, "
                                  "but 'flow.flow_id' is not None.")
        with self.assertRaisesRegex(openml.exceptions.PyOpenMLError, expected_message_regex):
            openml.runs.run_flow_on_task(
                task=task,
                flow=flow,
                avoid_duplicate_runs=True,
            )

    def test_run_with_illegal_flow_id_after_load(self):
        # Same as `test_run_with_illegal_flow_id`, but test this error is also
        # caught if the run is stored to and loaded from disk first.
        task = openml.tasks.get_task(115)
        clf = DecisionTreeClassifier()
        flow = self.extension.model_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        flow.flow_id = -1
        run = openml.runs.run_flow_on_task(
            task=task,
            flow=flow,
            avoid_duplicate_runs=False,
            upload_flow=False
        )

        cache_path = os.path.join(
            self.workdir,
            'runs',
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)
        loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)

        expected_message_regex = ("Flow does not exist on the server, "
                                  "but 'flow.flow_id' is not None.")
        with self.assertRaisesRegex(openml.exceptions.PyOpenMLError, expected_message_regex):
            loaded_run.publish()
            TestBase._mark_entity_for_removal('run', loaded_run.run_id)
            TestBase.logger.info("collected from test_run_functions: {}".format(loaded_run.run_id))

    def test_run_with_illegal_flow_id_1(self):
        # Check the case where the user adds an illegal flow id to an existing
        # flow. Comes to a different value error than the previous test
        task = openml.tasks.get_task(115)
        clf = DecisionTreeClassifier()
        flow_orig = self.extension.model_to_flow(clf)
        try:
            flow_orig.publish()  # ensures flow exist on server
            TestBase._mark_entity_for_removal('flow', (flow_orig.flow_id, flow_orig.name))
            TestBase.logger.info("collected from test_run_functions: {}".format(flow_orig.flow_id))
        except openml.exceptions.OpenMLServerException:
            # flow already exists
            pass
        flow_new = self.extension.model_to_flow(clf)

        flow_new.flow_id = -1
        expected_message_regex = (
            "Local flow_id does not match server flow_id: "
            "'-1' vs '[0-9]+'"
        )
        with self.assertRaisesRegex(openml.exceptions.PyOpenMLError, expected_message_regex):
            openml.runs.run_flow_on_task(
                task=task,
                flow=flow_new,
                avoid_duplicate_runs=True,
            )

    def test_run_with_illegal_flow_id_1_after_load(self):
        # Same as `test_run_with_illegal_flow_id_1`, but test this error is
        # also caught if the run is stored to and loaded from disk first.
        task = openml.tasks.get_task(115)
        clf = DecisionTreeClassifier()
        flow_orig = self.extension.model_to_flow(clf)
        try:
            flow_orig.publish()  # ensures flow exist on server
            TestBase._mark_entity_for_removal('flow', (flow_orig.flow_id, flow_orig.name))
            TestBase.logger.info("collected from test_run_functions: {}".format(flow_orig.flow_id))
        except openml.exceptions.OpenMLServerException:
            # flow already exists
            pass
        flow_new = self.extension.model_to_flow(clf)
        flow_new.flow_id = -1

        run = openml.runs.run_flow_on_task(
            task=task,
            flow=flow_new,
            avoid_duplicate_runs=False,
            upload_flow=False
        )

        cache_path = os.path.join(
            self.workdir,
            'runs',
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)
        loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)

        expected_message_regex = (
            "Local flow_id does not match server flow_id: "
            "'-1' vs '[0-9]+'"
        )
        self.assertRaisesRegex(
            openml.exceptions.PyOpenMLError,
            expected_message_regex,
            loaded_run.publish
        )

    def test__run_task_get_arffcontent(self):
        task = openml.tasks.get_task(7)
        num_instances = 3196
        num_folds = 10
        num_repeats = 1

        flow = unittest.mock.Mock()
        flow.name = 'dummy'
        clf = SGDClassifier(loss='log', random_state=1)
        res = openml.runs.functions._run_task_get_arffcontent(
            flow=flow,
            extension=self.extension,
            model=clf,
            task=task,
            add_local_measures=True,
        )
        arff_datacontent, trace, fold_evaluations, _ = res
        # predictions
        self.assertIsInstance(arff_datacontent, list)
        # trace. SGD does not produce any
        self.assertIsInstance(trace, type(None))

        task_type = TaskTypeEnum.SUPERVISED_CLASSIFICATION
        self._check_fold_timing_evaluations(fold_evaluations, num_repeats, num_folds,
                                            task_type=task_type)

        # 10 times 10 fold CV of 150 samples
        self.assertEqual(len(arff_datacontent), num_instances * num_repeats)
        for arff_line in arff_datacontent:
            # check number columns
            self.assertEqual(len(arff_line), 8)
            # check repeat
            self.assertGreaterEqual(arff_line[0], 0)
            self.assertLessEqual(arff_line[0], num_repeats - 1)
            # check fold
            self.assertGreaterEqual(arff_line[1], 0)
            self.assertLessEqual(arff_line[1], num_folds - 1)
            # check row id
            self.assertGreaterEqual(arff_line[2], 0)
            self.assertLessEqual(arff_line[2], num_instances - 1)
            # check confidences
            self.assertAlmostEqual(sum(arff_line[4:6]), 1.0)
            self.assertIn(arff_line[6], ['won', 'nowin'])
            self.assertIn(arff_line[7], ['won', 'nowin'])

    def test__create_trace_from_arff(self):
        with open(self.static_cache_dir + '/misc/trace.arff',
                  'r') as arff_file:
            trace_arff = arff.load(arff_file)
        OpenMLRunTrace.trace_from_arff(trace_arff)

    def test_get_run(self):
        # this run is not available on test
        openml.config.server = self.production_server
        run = openml.runs.get_run(473351)
        self.assertEqual(run.dataset_id, 357)
        self.assertEqual(run.evaluations['f_measure'], 0.841225)
        for i, value in [(0, 0.840918),
                         (1, 0.839458),
                         (2, 0.839613),
                         (3, 0.842571),
                         (4, 0.839567),
                         (5, 0.840922),
                         (6, 0.840985),
                         (7, 0.847129),
                         (8, 0.84218),
                         (9, 0.844014)]:
            self.assertEqual(run.fold_evaluations['f_measure'][0][i], value)
        assert ('weka' in run.tags)
        assert ('weka_3.7.12' in run.tags)
        assert (
            run.predictions_url == (
                "https://www.openml.org/data/download/1667125/"
                "weka_generated_predictions4575715871712251329.arff"
            )
        )

    def _check_run(self, run):
        # This tests that the API returns seven entries for each run
        # Check out https://openml.org/api/v1/xml/run/list/flow/1154
        # They are run_id, task_id, task_type_id, setup_id, flow_id, uploader, upload_time
        # error_message and run_details exist, too, but are not used so far. We need to update
        # this check once they are used!
        self.assertIsInstance(run, dict)
        assert len(run) == 8, str(run)

    def test_get_runs_list(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        runs = openml.runs.list_runs(id=[2], show_errors=True)
        self.assertEqual(len(runs), 1)
        for rid in runs:
            self._check_run(runs[rid])

    def test_list_runs_empty(self):
        runs = openml.runs.list_runs(task=[0])
        if len(runs) > 0:
            raise ValueError('UnitTest Outdated, got somehow results')

        self.assertIsInstance(runs, dict)

    def test_list_runs_output_format(self):
        runs = openml.runs.list_runs(size=1000, output_format='dataframe')
        self.assertIsInstance(runs, pd.DataFrame)

    def test_get_runs_list_by_task(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        task_ids = [20]
        runs = openml.runs.list_runs(task=task_ids)
        self.assertGreaterEqual(len(runs), 590)
        for rid in runs:
            self.assertIn(runs[rid]['task_id'], task_ids)
            self._check_run(runs[rid])
        num_runs = len(runs)

        task_ids.append(21)
        runs = openml.runs.list_runs(task=task_ids)
        self.assertGreaterEqual(len(runs), num_runs + 1)
        for rid in runs:
            self.assertIn(runs[rid]['task_id'], task_ids)
            self._check_run(runs[rid])

    def test_get_runs_list_by_uploader(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        # 29 is Dominik Kirchhoff
        uploader_ids = [29]

        runs = openml.runs.list_runs(uploader=uploader_ids)
        self.assertGreaterEqual(len(runs), 2)
        for rid in runs:
            self.assertIn(runs[rid]['uploader'], uploader_ids)
            self._check_run(runs[rid])
        num_runs = len(runs)

        uploader_ids.append(274)

        runs = openml.runs.list_runs(uploader=uploader_ids)
        self.assertGreaterEqual(len(runs), num_runs + 1)
        for rid in runs:
            self.assertIn(runs[rid]['uploader'], uploader_ids)
            self._check_run(runs[rid])

    def test_get_runs_list_by_flow(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        flow_ids = [1154]
        runs = openml.runs.list_runs(flow=flow_ids)
        self.assertGreaterEqual(len(runs), 1)
        for rid in runs:
            self.assertIn(runs[rid]['flow_id'], flow_ids)
            self._check_run(runs[rid])
        num_runs = len(runs)

        flow_ids.append(1069)
        runs = openml.runs.list_runs(flow=flow_ids)
        self.assertGreaterEqual(len(runs), num_runs + 1)
        for rid in runs:
            self.assertIn(runs[rid]['flow_id'], flow_ids)
            self._check_run(runs[rid])

    def test_get_runs_pagination(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        uploader_ids = [1]
        size = 10
        max = 100
        for i in range(0, max, size):
            runs = openml.runs.list_runs(offset=i, size=size,
                                         uploader=uploader_ids)
            self.assertGreaterEqual(size, len(runs))
            for rid in runs:
                self.assertIn(runs[rid]["uploader"], uploader_ids)

    def test_get_runs_list_by_filters(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        ids = [505212, 6100]
        tasks = [2974, 339]
        uploaders_1 = [1, 2]
        uploaders_2 = [29, 274]
        flows = [74, 1718]

        '''
        Since the results are taken by batch size, the function does not
        throw an OpenMLServerError anymore. Instead it throws a
        TimeOutException. For the moment commented out.
        '''
        # self.assertRaises(openml.exceptions.OpenMLServerError,
        # openml.runs.list_runs)

        runs = openml.runs.list_runs(id=ids)
        self.assertEqual(len(runs), 2)

        runs = openml.runs.list_runs(task=tasks)
        self.assertGreaterEqual(len(runs), 2)

        runs = openml.runs.list_runs(uploader=uploaders_2)
        self.assertGreaterEqual(len(runs), 10)

        runs = openml.runs.list_runs(flow=flows)
        self.assertGreaterEqual(len(runs), 100)

        runs = openml.runs.list_runs(id=ids, task=tasks, uploader=uploaders_1)

    def test_get_runs_list_by_tag(self):
        # TODO: comes from live, no such lists on test
        # Unit test works on production server only
        openml.config.server = self.production_server
        runs = openml.runs.list_runs(tag='curves')
        self.assertGreaterEqual(len(runs), 1)

    def test_run_on_dataset_with_missing_labels(self):
        # Check that _run_task_get_arffcontent works when one of the class
        # labels only declared in the arff file, but is not present in the
        # actual data

        flow = unittest.mock.Mock()
        flow.name = 'dummy'
        task = openml.tasks.get_task(2)

        model = Pipeline(steps=[('Imputer', SimpleImputer(strategy='median')),
                                ('Estimator', DecisionTreeClassifier())])

        data_content, _, _, _ = _run_task_get_arffcontent(
            flow=flow,
            model=model,
            task=task,
            extension=self.extension,
            add_local_measures=True,
        )
        # 2 folds, 5 repeats; keep in mind that this task comes from the test
        # server, the task on the live server is different
        self.assertEqual(len(data_content), 4490)
        for row in data_content:
            # repeat, fold, row_id, 6 confidences, prediction and correct label
            self.assertEqual(len(row), 12)

    def test_get_cached_run(self):
        openml.config.cache_directory = self.static_cache_dir
        openml.runs.functions._get_cached_run(1)

    def test_get_uncached_run(self):
        openml.config.cache_directory = self.static_cache_dir
        with self.assertRaises(openml.exceptions.OpenMLCacheException):
            openml.runs.functions._get_cached_run(10)

    def test_run_flow_on_task_downloaded_flow(self):
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=33)
        flow = self.extension.model_to_flow(model)
        flow.publish(raise_error_if_exists=False)
        TestBase._mark_entity_for_removal('flow', (flow.flow_id, flow.name))
        TestBase.logger.info("collected from test_run_functions: {}".format(flow.flow_id))

        downloaded_flow = openml.flows.get_flow(flow.flow_id)
        task = openml.tasks.get_task(119)  # diabetes
        run = openml.runs.run_flow_on_task(
            flow=downloaded_flow,
            task=task,
            avoid_duplicate_runs=False,
            upload_flow=False,
        )

        run.publish()
        TestBase._mark_entity_for_removal('run', run.run_id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split('/')[-1], run.run_id))
