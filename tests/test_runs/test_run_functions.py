import arff
import json
import random
import time
import sys

import numpy as np

import openml
import openml.exceptions
import openml._api_calls
import sklearn

from openml.testing import TestBase
from openml.runs.functions import _run_task_get_arffcontent, \
    _get_seeded_model, _run_exists, _extract_arfftrace, \
    _extract_arfftrace_attributes, _prediction_to_row, _check_n_jobs
from openml.flows.sklearn_converter import sklearn_to_flow

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection._search import BaseSearchCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.preprocessing.imputation import Imputer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, SGDClassifier, \
    LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, \
    StratifiedKFold
from sklearn.pipeline import Pipeline


class HardNaiveBayes(GaussianNB):
    # class for testing a naive bayes classifier that does not allow soft predictions
    def __init__(self, priors=None):
        super(HardNaiveBayes, self).__init__(priors)

    def predict_proba(*args, **kwargs):
        raise AttributeError('predict_proba is not available when  probability=False')


class TestRun(TestBase):
    _multiprocess_can_split_ = True

    def _wait_for_processed_run(self, run_id, max_waiting_time_seconds):
        # it can take a while for a run to be processed on the OpenML (test) server
        # however, sometimes it is good to wait (a bit) for this, to properly test
        # a function. In this case, we wait for max_waiting_time_seconds on this
        # to happen, probing the server every 10 seconds to speed up the process

        # time.time() works in seconds
        start_time = time.time()
        while time.time() - start_time < max_waiting_time_seconds:
            run = openml.runs.get_run(run_id)
            if len(run.evaluations) > 0:
                return
            else:
                time.sleep(10)

    def _check_serialized_optimized_run(self, run_id):
        run = openml.runs.get_run(run_id)
        task = openml.tasks.get_task(run.task_id)

        # TODO: assert holdout task

        # downloads the predictions of the old task
        predictions_url = openml._api_calls._file_id_to_url(run.output_files['predictions'])
        predictions = arff.loads(openml._api_calls._read_url(predictions_url))

        # downloads the best model based on the optimization trace
        # suboptimal (slow), and not guaranteed to work if evaluation
        # engine is behind. TODO: mock this? We have the arff already on the server
        self._wait_for_processed_run(run_id, 200)
        try:
            model_prime = openml.runs.initialize_model_from_trace(run_id, 0, 0)
        except openml.exceptions.OpenMLServerException as e:
            e.additional = str(e.additional) + '; run_id: ' + str(run_id)
            raise e
        
        run_prime = openml.runs.run_model_on_task(task, model_prime,
                                                  avoid_duplicate_runs=False,
                                                  seed=1)
        predictions_prime = run_prime._generate_arff_dict()

        self.assertEquals(len(predictions_prime['data']), len(predictions['data']))

        # The original search model does not submit confidence bounds,
        # so we can not compare the arff line
        compare_slice = [0, 1, 2, -1, -2]
        for idx in range(len(predictions['data'])):
            # depends on the assumption "predictions are in same order"
            # that does not necessarily hold.
            # But with the current code base, it holds.
            for col_idx in compare_slice:
                self.assertEquals(predictions['data'][idx][col_idx], predictions_prime['data'][idx][col_idx])

        return True

    def _perform_run(self, task_id, num_instances, clf,
                     random_state_value=None, check_setup=True):

        def _remove_random_state(flow):
            if 'random_state' in flow.parameters:
                del flow.parameters['random_state']
            for component in flow.components.values():
                _remove_random_state(component)

        flow = sklearn_to_flow(clf)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        flow.publish()

        task = openml.tasks.get_task(task_id)
        run = openml.runs.run_flow_on_task(task, flow, seed=1,
                                           avoid_duplicate_runs=openml.config.avoid_duplicate_runs)
        run_ = run.publish()
        self.assertEqual(run_, run)
        self.assertIsInstance(run.dataset_id, int)

        # check arff output
        self.assertEqual(len(run.data_content), num_instances)

        if check_setup:
            # test the initialize setup function
            run_id = run_.run_id
            run_server = openml.runs.get_run(run_id)
            clf_server = openml.setups.initialize_model(run_server.setup_id)

            flow_local = openml.flows.sklearn_to_flow(clf)
            flow_server = openml.flows.sklearn_to_flow(clf_server)

            if flow.class_name not in \
                    ['sklearn.model_selection._search.GridSearchCV',
                     'sklearn.pipeline.Pipeline']:
                # If the flow is initialized from a model without a random state,
                # the flow is on the server without any random state
                self.assertEqual(flow.parameters['random_state'], 'null')
                # As soon as a flow is run, a random state is set in the model.
                # If a flow is re-instantiated
                self.assertEqual(flow_local.parameters['random_state'],
                                 random_state_value)
                self.assertEqual(flow_server.parameters['random_state'],
                                 random_state_value)
            _remove_random_state(flow_local)
            _remove_random_state(flow_server)
            openml.flows.assert_flows_equal(flow_local, flow_server)

            # and test the initialize setup from run function
            clf_server2 = openml.runs.initialize_model_from_run(run_server.run_id)
            flow_server2 = openml.flows.sklearn_to_flow(clf_server2)
            if flow.class_name not in \
                    ['sklearn.model_selection._search.GridSearchCV',
                     'sklearn.pipeline.Pipeline']:
                self.assertEqual(flow_server2.parameters['random_state'],
                                 random_state_value)

            _remove_random_state(flow_server2)
            openml.flows.assert_flows_equal(flow_local, flow_server2)

            #self.assertEquals(clf.get_params(), clf_prime.get_params())
            # self.assertEquals(clf, clf_prime)

        downloaded = openml.runs.get_run(run_.run_id)
        assert('openml-python' in downloaded.tags)

        return run

    def _check_fold_evaluations(self, fold_evaluations, num_repeats, num_folds, max_time_allowed=60000):
        '''
        Checks whether the right timing measures are attached to the run (before upload).
        Test is only performed for versions >= Python3.3

        In case of check_n_jobs(clf) == false, please do not perform this check (check this
        condition outside of this function. )
        default max_time_allowed (per fold, in milli seconds) = 1 minute, quite pessimistic
        '''

        # a dict mapping from openml measure to a tuple with the minimum and maximum allowed value
        check_measures = {'usercpu_time_millis_testing': (0, max_time_allowed),
                          'usercpu_time_millis_training': (0, max_time_allowed),  # should take at least one millisecond (?)
                          'usercpu_time_millis': (0, max_time_allowed),
                          'predictive_accuracy': (0, 1)}

        self.assertIsInstance(fold_evaluations, dict)
        if sys.version_info[:2] >= (3, 3):
            # this only holds if we are allowed to record time (otherwise some are missing)
            self.assertEquals(set(fold_evaluations.keys()), set(check_measures.keys()))

        for measure in check_measures.keys():
            if measure in fold_evaluations:
                num_rep_entrees = len(fold_evaluations[measure])
                self.assertEquals(num_rep_entrees, num_repeats)
                min_val = check_measures[measure][0]
                max_val = check_measures[measure][1]
                for rep in range(num_rep_entrees):
                    num_fold_entrees = len(fold_evaluations[measure][rep])
                    self.assertEquals(num_fold_entrees, num_folds)
                    for fold in range(num_fold_entrees):
                        evaluation = fold_evaluations[measure][rep][fold]
                        self.assertIsInstance(evaluation, float)
                        self.assertGreaterEqual(evaluation, min_val)
                        self.assertLessEqual(evaluation, max_val)


    def _check_sample_evaluations(self, sample_evaluations, num_repeats, num_folds, num_samples, max_time_allowed=60000):
        '''
        Checks whether the right timing measures are attached to the run (before upload).
        Test is only performed for versions >= Python3.3

        In case of check_n_jobs(clf) == false, please do not perform this check (check this
        condition outside of this function. )
        default max_time_allowed (per fold, in milli seconds) = 1 minute, quite pessimistic
        '''

        # a dict mapping from openml measure to a tuple with the minimum and maximum allowed value
        check_measures = {'usercpu_time_millis_testing': (0, max_time_allowed),
                          'usercpu_time_millis_training': (0, max_time_allowed),  # should take at least one millisecond (?)
                          'usercpu_time_millis': (0, max_time_allowed),
                          'predictive_accuracy': (0, 1)}

        self.assertIsInstance(sample_evaluations, dict)
        if sys.version_info[:2] >= (3, 3):
            # this only holds if we are allowed to record time (otherwise some are missing)
            self.assertEquals(set(sample_evaluations.keys()), set(check_measures.keys()))

        for measure in check_measures.keys():
            if measure in sample_evaluations:
                num_rep_entrees = len(sample_evaluations[measure])
                self.assertEquals(num_rep_entrees, num_repeats)
                for rep in range(num_rep_entrees):
                    num_fold_entrees = len(sample_evaluations[measure][rep])
                    self.assertEquals(num_fold_entrees, num_folds)
                    for fold in range(num_fold_entrees):
                        num_sample_entrees = len(sample_evaluations[measure][rep][fold])
                        self.assertEquals(num_sample_entrees, num_samples)
                        for sample in range(num_sample_entrees):
                            evaluation = sample_evaluations[measure][rep][fold][sample]
                            self.assertIsInstance(evaluation, float)
                            self.assertGreater(evaluation, 0) # should take at least one millisecond (?)
                            self.assertLess(evaluation, max_time_allowed)

    def test_run_regression_on_classif_task(self):
        task_id = 115

        clf = LinearRegression()
        task = openml.tasks.get_task(task_id)
        self.assertRaises(AttributeError, openml.runs.run_model_on_task,
                          task=task, model=clf, avoid_duplicate_runs=False)

    def test_check_erronous_sklearn_flow_fails(self):
        task_id = 115
        task = openml.tasks.get_task(task_id)

        # Invalid parameter values
        clf = LogisticRegression(C='abc')
        self.assertRaisesRegexp(ValueError,
                                "Penalty term must be positive; got "
                                # u? for 2.7/3.4-6 compability
                                "\(C=u?'abc'\)",
                                openml.runs.run_model_on_task, task=task,
                                model=clf)

    def test__publish_flow_if_necessary(self):
        task_id = 115
        task = openml.tasks.get_task(task_id)

        clf = LogisticRegression()
        flow = sklearn_to_flow(clf)
        flow, sentinel = self._add_sentinel_to_flow_name(flow, None)
        openml.runs.functions._publish_flow_if_necessary(flow)
        self.assertIsNotNone(flow.flow_id)

        flow2 = sklearn_to_flow(clf)
        flow2, _ = self._add_sentinel_to_flow_name(flow2, sentinel)
        openml.runs.functions._publish_flow_if_necessary(flow2)
        self.assertEqual(flow2.flow_id, flow.flow_id)

    ############################################################################
    # These unit tests are ment to test the following functions, using a varity
    #  of flows:
    # - openml.runs.run_task()
    # - openml.runs.OpenMLRun.publish()
    # - openml.runs.initialize_model()
    # - [implicitly] openml.setups.initialize_model()
    # - openml.runs.initialize_model_from_trace()
    # They're split among several actual functions to allow for parallel
    # execution of the unit tests without the need to add an additional module
    # like unittest2

    def _run_and_upload(self, clf, rsv):
        task_id = 119  # diabates dataset
        num_test_instances = 253  # 33% holdout task
        num_folds = 1  # because of holdout
        num_iterations = 5  # for base search classifiers

        run = self._perform_run(task_id, num_test_instances, clf,
                                random_state_value=rsv)

        # obtain accuracy scores using get_metric_score:
        accuracy_scores = run.get_metric_fn(sklearn.metrics.accuracy_score)
        # compare with the scores in user defined measures
        accuracy_scores_provided = []
        for rep in run.fold_evaluations['predictive_accuracy'].keys():
            for fold in run.fold_evaluations['predictive_accuracy'][rep].keys():
                accuracy_scores_provided.append(
                    run.fold_evaluations['predictive_accuracy'][rep][fold])
        self.assertEquals(sum(accuracy_scores_provided), sum(accuracy_scores))

        if isinstance(clf, BaseSearchCV):
            if isinstance(clf, GridSearchCV):
                grid_iterations = 1
                for param in clf.param_grid:
                    grid_iterations *= len(clf.param_grid[param])
                self.assertEqual(len(run.trace_content),
                                 grid_iterations * num_folds)
            else:
                self.assertEqual(len(run.trace_content),
                                 num_iterations * num_folds)
            check_res = self._check_serialized_optimized_run(run.run_id)
            self.assertTrue(check_res)

        # todo: check if runtime is present
        self._check_fold_evaluations(run.fold_evaluations, 1, num_folds)
        pass

    def test_run_and_upload_logistic_regression(self):
        lr = LogisticRegression()
        self._run_and_upload(lr, '62501')

    def test_run_and_upload_pipeline_dummy_pipeline(self):

        pipeline1 = Pipeline(steps=[('scaler', StandardScaler(with_mean=False)),
                                    ('dummy', DummyClassifier(strategy='prior'))])
        self._run_and_upload(pipeline1, '62501')

    def test_run_and_upload_decision_tree_pipeline(self):
        pipeline2 = Pipeline(steps=[('Imputer', Imputer(strategy='median')),
                                    ('VarianceThreshold', VarianceThreshold()),
                                    ('Estimator', RandomizedSearchCV(
                                        DecisionTreeClassifier(),
                                        {'min_samples_split': [2 ** x for x in range(1, 7 + 1)],
                                         'min_samples_leaf': [2 ** x for x in range(0, 6 + 1)]},
                                        cv=3, n_iter=10))])
        self._run_and_upload(pipeline2, '62501')

    def test_run_and_upload_gridsearch(self):
        gridsearch = GridSearchCV(BaggingClassifier(base_estimator=SVC()),
                                  {"base_estimator__C": [0.01, 0.1, 10],
                                   "base_estimator__gamma": [0.01, 0.1, 10]})
        self._run_and_upload(gridsearch, '62501')

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
        self._run_and_upload(randomsearch, '12172')

    ############################################################################

    def test_learning_curve_task_1(self):
        task_id = 801  # diabates dataset
        num_test_instances = 6144 # for learning curve
        num_repeats = 1
        num_folds = 10
        num_samples = 8

        pipeline1 = Pipeline(steps=[('scaler', StandardScaler(with_mean=False)),
                                    ('dummy', DummyClassifier(strategy='prior'))])
        run = self._perform_run(task_id, num_test_instances, pipeline1,
                                random_state_value='62501')
        self._check_sample_evaluations(run.sample_evaluations, num_repeats,
                                       num_folds, num_samples)

    def test_learning_curve_task_2(self):
        task_id = 801  # diabates dataset
        num_test_instances = 6144  # for learning curve
        num_repeats = 1
        num_folds = 10
        num_samples = 8

        pipeline2 = Pipeline(steps=[('Imputer', Imputer(strategy='median')),
                                    ('VarianceThreshold', VarianceThreshold()),
                                    ('Estimator', RandomizedSearchCV(
                                        DecisionTreeClassifier(),
                                        {'min_samples_split': [2 ** x for x in range(1, 7 + 1)],
                                         'min_samples_leaf': [2 ** x for x in range(0, 6 + 1)]},
                                        cv=3, n_iter=10))])
        run = self._perform_run(task_id, num_test_instances, pipeline2,
                                random_state_value='62501')
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
        run = openml.runs.run_model_on_task(task, randomsearch,
                                            avoid_duplicate_runs=False, seed=1)
        run_ = run.publish()
        run = openml.runs.get_run(run_.run_id)

        modelR = openml.runs.initialize_model_from_run(run.run_id)
        modelS = openml.setups.initialize_model(run.setup_id)

        self.assertEquals(modelS.cv.random_state, 62501)
        self.assertEqual(modelR.cv.random_state, 62501)

    def _test_local_evaluations(self, run):

        # compare with the scores in user defined measures
        accuracy_scores_provided = []
        for rep in run.fold_evaluations['predictive_accuracy'].keys():
            for fold in run.fold_evaluations['predictive_accuracy'][rep].keys():
                accuracy_scores_provided.append(run.fold_evaluations['predictive_accuracy'][rep][fold])
        accuracy_scores = run.get_metric_fn(sklearn.metrics.accuracy_score)
        np.testing.assert_array_almost_equal(accuracy_scores_provided, accuracy_scores)

        # also check if we can obtain some other scores: # TODO: how to do AUC?
        tests = [(sklearn.metrics.cohen_kappa_score, {'weights': None}),
                 (sklearn.metrics.auc, {'reorder': True}),
                 (sklearn.metrics.average_precision_score, {}),
                 (sklearn.metrics.jaccard_similarity_score, {}),
                 (sklearn.metrics.precision_score, {'average': 'macro'}),
                 (sklearn.metrics.brier_score_loss, {})]
        for test_idx, test in enumerate(tests):
            alt_scores = run.get_metric_fn(test[0], test[1])
            self.assertEquals(len(alt_scores), 10)
            for idx in range(len(alt_scores)):
                self.assertGreaterEqual(alt_scores[idx], 0)
                self.assertLessEqual(alt_scores[idx], 1)

    def test_local_run_metric_score(self):

        # construct sci-kit learn classifier
        clf = Pipeline(steps=[('imputer', Imputer(strategy='median')), ('estimator', RandomForestClassifier())])

        # download task
        task = openml.tasks.get_task(7)

        # invoke OpenML run
        run = openml.runs.run_model_on_task(task, clf)

        self._test_local_evaluations(run)

    def test_online_run_metric_score(self):
        openml.config.server = self.production_server
        run = openml.runs.get_run(5965513) # important to use binary classification task, due to assertions
        self._test_local_evaluations(run)

    def test_initialize_model_from_run(self):
        clf = sklearn.pipeline.Pipeline(steps=[('Imputer', Imputer(strategy='median')),
                                               ('VarianceThreshold', VarianceThreshold(threshold=0.05)),
                                               ('Estimator', GaussianNB())])
        task = openml.tasks.get_task(11)
        run = openml.runs.run_model_on_task(task, clf, avoid_duplicate_runs=False)
        run_ = run.publish()
        run = openml.runs.get_run(run_.run_id)

        modelR = openml.runs.initialize_model_from_run(run.run_id)
        modelS = openml.setups.initialize_model(run.setup_id)

        flowR = openml.flows.sklearn_to_flow(modelR)
        flowS = openml.flows.sklearn_to_flow(modelS)
        flowL = openml.flows.sklearn_to_flow(clf)
        openml.flows.assert_flows_equal(flowR, flowL)
        openml.flows.assert_flows_equal(flowS, flowL)

        self.assertEquals(flowS.components['Imputer'].parameters['strategy'], '"median"')
        self.assertEquals(flowS.components['VarianceThreshold'].parameters['threshold'], '0.05')

    def test_get_run_trace(self):
        # get_run_trace is already tested implicitly in test_run_and_publish
        # this test is a bit additional.
        num_iterations = 10
        num_folds = 1
        task_id = 119

        task = openml.tasks.get_task(task_id)
        # IMPORTANT! Do not sentinel this flow. is faster if we don't wait on openml server
        clf = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                 {"max_depth": [3, None],
                                  "max_features": [1, 2, 3, 4],
                                  "bootstrap": [True, False],
                                  "criterion": ["gini", "entropy"]},
                                 num_iterations, random_state=42)

        # [SPEED] make unit test faster by exploiting run information from the past
        try:
            # in case the run did not exists yet
            run = openml.runs.run_model_on_task(task, clf, avoid_duplicate_runs=True)
            trace = openml.runs.functions._create_trace_from_arff(run._generate_trace_arff_dict())
            self.assertEquals(
                len(trace.trace_iterations),
                num_iterations * num_folds,
            )
            run = run.publish()
            self._wait_for_processed_run(run.run_id, 200)
            run_id = run.run_id
        except openml.exceptions.PyOpenMLError:
            # run was already
            flow = openml.flows.sklearn_to_flow(clf)
            flow_exists = openml.flows.flow_exists(flow.name, flow.external_version)
            self.assertIsInstance(flow_exists, int)
            downloaded_flow = openml.flows.get_flow(flow_exists)
            setup_exists = openml.setups.setup_exists(downloaded_flow)
            self.assertIsInstance(setup_exists, int)
            run_ids = _run_exists(task.task_id, setup_exists)
            run_id = random.choice(list(run_ids))

        # now the actual unit test ...
        run_trace = openml.runs.get_run_trace(run_id)
        self.assertEqual(len(run_trace.trace_iterations), num_iterations * num_folds)

    def test__run_exists(self):
        # would be better to not sentinel these clfs,
        # so we do not have to perform the actual runs
        # and can just check their status on line
        clfs = [sklearn.pipeline.Pipeline(steps=[('Imputer', Imputer(strategy='mean')),
                                                ('VarianceThreshold', VarianceThreshold(threshold=0.05)),
                                                ('Estimator', DecisionTreeClassifier(max_depth=4))]),
                sklearn.pipeline.Pipeline(steps=[('Imputer', Imputer(strategy='most_frequent')),
                                                 ('VarianceThreshold', VarianceThreshold(threshold=0.1)),
                                                 ('Estimator', DecisionTreeClassifier(max_depth=4))])]

        task = openml.tasks.get_task(115)

        for clf in clfs:
            try:
                # first populate the server with this run.
                # skip run if it was already performed.
                run = openml.runs.run_model_on_task(task, clf, avoid_duplicate_runs=True)
                run.publish()
            except openml.exceptions.PyOpenMLError as e:
                # run already existed. Great.
                pass

            flow = openml.flows.sklearn_to_flow(clf)
            flow_exists = openml.flows.flow_exists(flow.name, flow.external_version)
            self.assertGreater(flow_exists, 0)
            downloaded_flow = openml.flows.get_flow(flow_exists)
            setup_exists = openml.setups.setup_exists(downloaded_flow, clf)
            self.assertGreater(setup_exists, 0)
            run_ids = _run_exists(task.task_id, setup_exists)
            self.assertTrue(run_ids, msg=(run_ids, clf))

    def test__get_seeded_model(self):
        # randomized models that are initialized without seeds, can be seeded
        randomized_clfs = [
            BaggingClassifier(),
            RandomizedSearchCV(RandomForestClassifier(),
                               {"max_depth": [3, None],
                                "max_features": [1, 2, 3, 4],
                                "bootstrap": [True, False],
                                "criterion": ["gini", "entropy"],
                                "random_state" : [-1, 0, 1, 2]},
                               cv=StratifiedKFold(n_splits=2, shuffle=True)),
            DummyClassifier()
        ]

        for idx, clf in enumerate(randomized_clfs):
            const_probe = 42
            all_params = clf.get_params()
            params = [key for key in all_params if key.endswith('random_state')]
            self.assertGreater(len(params), 0)

            # before param value is None
            for param in params:
                self.assertIsNone(all_params[param])

            # now seed the params
            clf_seeded = _get_seeded_model(clf, const_probe)
            new_params = clf_seeded.get_params()

            randstate_params = [key for key in new_params if key.endswith('random_state')]

            # afterwards, param value is set
            for param in randstate_params:
                self.assertIsInstance(new_params[param], int)
                self.assertIsNotNone(new_params[param])

            if idx == 1:
                self.assertEqual(clf.cv.random_state, 56422)

    def test__get_seeded_model_raises(self):
        # the _get_seeded_model should raise exception if random_state is anything else than an int
        randomized_clfs = [
            BaggingClassifier(random_state=np.random.RandomState(42)),
            DummyClassifier(random_state="OpenMLIsGreat")
        ]

        for clf in randomized_clfs:
            self.assertRaises(ValueError, _get_seeded_model, model=clf, seed=42)

    def test__extract_arfftrace(self):
        param_grid = {"max_depth": [3, None],
                      "max_features": [1, 2, 3, 4],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        num_iters = 10
        task = openml.tasks.get_task(20)
        clf = RandomizedSearchCV(RandomForestClassifier(), param_grid, num_iters)
        # just run the task
        train, _ = task.get_train_test_split_indices(0, 0)
        X, y = task.get_X_and_y()
        clf.fit(X[train], y[train])

        trace_attribute_list = _extract_arfftrace_attributes(clf)
        trace_list = _extract_arfftrace(clf, 0, 0)
        self.assertIsInstance(trace_attribute_list, list)
        self.assertEquals(len(trace_attribute_list), 5 + len(param_grid))
        self.assertIsInstance(trace_list, list)
        self.assertEquals(len(trace_list), num_iters)

        # found parameters
        optimized_params = set()

        for att_idx in range(len(trace_attribute_list)):
            att_type = trace_attribute_list[att_idx][1]
            att_name = trace_attribute_list[att_idx][0]
            if att_name.startswith("parameter_"):
                # add this to the found parameters
                param_name = att_name[len("parameter_"):]
                optimized_params.add(param_name)

                for line_idx in range(len(trace_list)):
                    val = json.loads(trace_list[line_idx][att_idx])
                    legal_values = param_grid[param_name]
                    self.assertIn(val, legal_values)
            else:
                # repeat, fold, itt, bool
                for line_idx in range(len(trace_list)):
                    val = trace_list[line_idx][att_idx]
                    if isinstance(att_type, list):
                        self.assertIn(val, att_type)
                    elif att_name in ['repeat', 'fold', 'iteration']:
                        self.assertIsInstance(trace_list[line_idx][att_idx], int)
                    else: # att_type = real
                        self.assertIsInstance(trace_list[line_idx][att_idx], float)


        self.assertEqual(set(param_grid.keys()), optimized_params)

    def test__prediction_to_row(self):
        repeat_nr = 0
        fold_nr = 0
        clf = sklearn.pipeline.Pipeline(steps=[('Imputer', Imputer(strategy='mean')),
                                               ('VarianceThreshold', VarianceThreshold(threshold=0.05)),
                                               ('Estimator', GaussianNB())])
        task = openml.tasks.get_task(20)
        train, test = task.get_train_test_split_indices(repeat_nr, fold_nr)
        X, y = task.get_X_and_y()
        clf.fit(X[train], y[train])

        test_X = X[test]
        test_y = y[test]

        probaY = clf.predict_proba(test_X)
        predY = clf.predict(test_X)
        sample_nr = 0 # default for this task
        for idx in range(0, len(test_X)):
            arff_line = _prediction_to_row(repeat_nr, fold_nr, sample_nr, idx,
                                           task.class_labels[test_y[idx]],
                                           predY[idx], probaY[idx], task.class_labels, clf.classes_)

            self.assertIsInstance(arff_line, list)
            self.assertEqual(len(arff_line), 6 + len(task.class_labels))
            self.assertEqual(arff_line[0], repeat_nr)
            self.assertEqual(arff_line[1], fold_nr)
            self.assertEqual(arff_line[2], sample_nr)
            self.assertEqual(arff_line[3], idx)
            sum = 0.0
            for att_idx in range(4, 4 + len(task.class_labels)):
                self.assertIsInstance(arff_line[att_idx], float)
                self.assertGreaterEqual(arff_line[att_idx], 0.0)
                self.assertLessEqual(arff_line[att_idx], 1.0)
                sum += arff_line[att_idx]
            self.assertAlmostEqual(sum, 1.0)

            self.assertIn(arff_line[-1], task.class_labels)
            self.assertIn(arff_line[-2], task.class_labels)
        pass

    def test_run_with_classifiers_in_param_grid(self):
        task = openml.tasks.get_task(115)

        param_grid = {
            "base_estimator": [DecisionTreeClassifier(), ExtraTreeClassifier()]
        }

        clf = GridSearchCV(BaggingClassifier(), param_grid=param_grid)
        self.assertRaises(TypeError, openml.runs.run_model_on_task,
                          task=task, model=clf, avoid_duplicate_runs=False)

    def test__run_task_get_arffcontent(self):
        task = openml.tasks.get_task(7)
        class_labels = task.class_labels
        num_instances = 3196
        num_folds = 10
        num_repeats = 1

        clf = SGDClassifier(loss='log', random_state=1)
        res = openml.runs.functions._run_task_get_arffcontent(clf, task, class_labels)
        arff_datacontent, arff_tracecontent, _, fold_evaluations, sample_evaluations = res
        # predictions
        self.assertIsInstance(arff_datacontent, list)
        # trace. SGD does not produce any
        self.assertIsInstance(arff_tracecontent, type(None))

        self._check_fold_evaluations(fold_evaluations, num_repeats, num_folds)

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
        with open(self.static_cache_dir + '/misc/trace.arff', 'r') as arff_file:
            trace_arff = arff.load(arff_file)
        trace = openml.runs.functions._create_trace_from_arff(trace_arff)

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
        assert('weka' in run.tags)
        assert('weka_3.7.12' in run.tags)

    def _check_run(self, run):
        self.assertIsInstance(run, dict)
        self.assertEqual(len(run), 5)

    def test_get_runs_list(self):
        # TODO: comes from live, no such lists on test
        openml.config.server = self.production_server
        runs = openml.runs.list_runs(id=[2])
        self.assertEqual(len(runs), 1)
        for rid in runs:
            self._check_run(runs[rid])

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
        # 29 is Dominik Kirchhoff - Joaquin and Jan have too many runs right now
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
            runs = openml.runs.list_runs(offset=i, size=size, uploader=uploader_ids)
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

        self.assertRaises(openml.exceptions.OpenMLServerError, openml.runs.list_runs)

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
        openml.config.server = self.production_server
        runs = openml.runs.list_runs(tag='curves')
        self.assertGreaterEqual(len(runs), 1)

    def test_run_on_dataset_with_missing_labels(self):
        # Check that _run_task_get_arffcontent works when one of the class
        # labels only declared in the arff file, but is not present in the
        # actual data

        task = openml.tasks.get_task(2)
        class_labels = task.class_labels

        model = Pipeline(steps=[('Imputer', Imputer(strategy='median')),
                                ('Estimator', DecisionTreeClassifier())])

        data_content, _, _, _, _ = _run_task_get_arffcontent(model, task, class_labels)
        # 2 folds, 5 repeats; keep in mind that this task comes from the test
        # server, the task on the live server is different
        self.assertEqual(len(data_content), 4490)
        for row in data_content:
            # repeat, fold, row_id, 6 confidences, prediction and correct label
            self.assertEqual(len(row), 12)

    def test_predict_proba_hardclassifier(self):
        # task 1 (test server) is important, as it is a task with an unused class
        tasks = [1, 3, 115]

        for task_id in tasks:
            task = openml.tasks.get_task(task_id)
            clf1 = sklearn.pipeline.Pipeline(steps=[
                ('imputer', sklearn.preprocessing.Imputer()), ('estimator', GaussianNB())
            ])
            clf2 = sklearn.pipeline.Pipeline(steps=[
                ('imputer', sklearn.preprocessing.Imputer()), ('estimator', HardNaiveBayes())
            ])

            arff_content1, arff_header1, _, _, _ = _run_task_get_arffcontent(clf1, task, task.class_labels)
            arff_content2, arff_header2, _, _, _ = _run_task_get_arffcontent(clf2, task, task.class_labels)

            # verifies last two arff indices (predict and correct)
            # TODO: programmatically check wether these are indeed features (predict, correct)
            predictionsA = np.array(arff_content1)[:, -2:]
            predictionsB = np.array(arff_content2)[:, -2:]

            np.testing.assert_array_equal(predictionsA, predictionsB)
