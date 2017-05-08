import sys
import arff
import time

import openml
import openml.exceptions
import openml._api_calls

from openml.testing import TestBase
from openml.runs.functions import _run_task_get_arffcontent

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing.imputation import Imputer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier, \
    LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, \
    StratifiedKFold
from sklearn.pipeline import Pipeline

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock


class TestRun(TestBase):

    def _check_serialized_optimized_run(self, run_id):
        run = openml.runs.get_run(run_id)
        task = openml.tasks.get_task(run.task_id)

        # TODO: assert holdout task

        # downloads the predictions of the old task
        predictions_url = openml._api_calls.fileid_to_url(run.output_files['predictions'])
        predictions = arff.loads(openml._api_calls._read_url(predictions_url))

        # downloads the best model based on the optimization trace
        # suboptimal (slow), and not guaranteed to work if evaluation
        # engine is behind. TODO: mock this? We have the arff already on the server
        secCount = 0
        while secCount < 70:
            try:
                model_prime = openml.runs.initialize_model_from_trace(run_id, 0, 0)
                break
            except openml.exceptions.OpenMLServerException:
                time.sleep(10)
                secCount += 10

        run_prime = openml.runs.run_task(task, model_prime, avoid_duplicate_runs=False)
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


    def _perform_run(self, task_id, num_instances, clf, check_setup=True):
        task = openml.tasks.get_task(task_id)
        run = openml.runs.run_task(task, clf, openml.config.avoid_duplicate_runs)
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

            openml.flows.assert_flows_equal(flow_local, flow_server)

            # and test the initialize setup from run function
            clf_server2 = openml.runs.initialize_model_from_run(run_server.run_id)
            flow_server2 = openml.flows.sklearn_to_flow(clf_server2)
            openml.flows.assert_flows_equal(flow_local, flow_server2)

            #self.assertEquals(clf.get_params(), clf_prime.get_params())
            # self.assertEquals(clf, clf_prime)

        return run

    def test_run_regression_on_classif_task(self):
        task_id = 115

        clf = LinearRegression()
        task = openml.tasks.get_task(task_id)
        self.assertRaises(AttributeError, openml.runs.run_task,
                          task=task, model=clf, avoid_duplicate_runs=False)

    @mock.patch('openml.flows.sklearn_to_flow')
    def test_check_erronous_sklearn_flow_fails(self, sklearn_to_flow_mock):
        task_id = 115
        task = openml.tasks.get_task(task_id)

        # Invalid parameter values
        clf = LogisticRegression(C='abc')
        self.assertEqual(sklearn_to_flow_mock.call_count, 0)
        self.assertRaisesRegexp(ValueError, "Penalty term must be positive; got \(C='abc'\)",
                                openml.runs.run_task, task=task, model=clf)

    def test_run_diabetes(self):
        task_id = 115
        num_instances = 768

        clf = LogisticRegression()
        res = self._perform_run(task_id,num_instances, clf)

        downloaded = openml.runs.get_run(res.run_id)
        assert('openml-python' in downloaded.tags)

    def test_run_optimize_randomforest_diabetes(self):
        task_id = 119
        num_test_instances = 253
        num_folds = 1
        num_iterations = 5

        clf = RandomForestClassifier(n_estimators=5)
        param_dist = {"max_depth": [3, None],
                      "max_features": [1,2,3,4],
                      "min_samples_split": [2,3,4,5,6,7,8,9,10],
                      "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        cv = StratifiedKFold(n_splits=3)
        random_search = RandomizedSearchCV(clf, param_dist, cv=cv,
                                           n_iter=num_iterations)

        run = self._perform_run(task_id, num_test_instances, random_search)
        self.assertEqual(len(run.trace_content), num_iterations * num_folds)

        res = self._check_serialized_optimized_run(run.run_id)
        self.assertTrue(res)

    def test_run_optimize_bagging_diabetes(self):
        task_id = 119
        num_test_instances = 253
        num_folds = 1
        num_iterations = 9 # (num values for C times gamma)

        bag = BaggingClassifier(base_estimator=SVC())
        param_dist = {"base_estimator__C": [0.01, 0.1, 10],
                      "base_estimator__gamma": [0.01, 0.1, 10]}
        grid_search = GridSearchCV(bag, param_dist)

        run = self._perform_run(task_id, num_test_instances, grid_search)
        self.assertEqual(len(run.trace_content), num_iterations * num_folds)

        res = self._check_serialized_optimized_run(run.run_id)
        self.assertTrue(res)

    def test_run_pipeline(self):
        task_id = 115
        num_instances = 768
        num_folds = 10
        num_iterations = 9  # (num values for C times gamma)

        scaler = StandardScaler(with_mean=False)
        dummy = DummyClassifier(strategy='prior')
        model = Pipeline(steps=(('scaler', scaler), ('dummy', dummy)))

        run = self._perform_run(task_id, num_instances, model)
        self.assertEqual(run.trace_content, None)

    def test__run_task_get_arffcontent(self):
        task = openml.tasks.get_task(7)
        class_labels = task.class_labels
        num_instances = 3196
        num_folds = 10
        num_repeats = 1

        clf = SGDClassifier(loss='hinge', random_state=1)
        self.assertRaisesRegexp(AttributeError,
                                "probability estimates are not available for loss='hinge'",
                                openml.runs.functions._run_task_get_arffcontent,
                                clf, task, class_labels)

        clf = SGDClassifier(loss='log', random_state=1)
        arff_datacontent, arff_tracecontent, _ = openml.runs.functions._run_task_get_arffcontent(
            clf, task, class_labels)
        # predictions
        self.assertIsInstance(arff_datacontent, list)
        # trace. SGD does not produce any
        self.assertIsInstance(arff_tracecontent, type(None))

        # 10 times 10 fold CV of 150 samples
        self.assertEqual(len(arff_datacontent), num_instances * num_repeats)
        for arff_line in arff_datacontent:
            # check number columns
            self.assertEqual(len(arff_line), 7)
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
            self.assertAlmostEqual(sum(arff_line[3:5]), 1.0)
            self.assertIn(arff_line[5], ['won', 'nowin'])
            self.assertIn(arff_line[6], ['won', 'nowin'])

    def test_get_run(self):
        # this run is not available on test
        openml.config.server = self.production_server
        run = openml.runs.get_run(473350)
        self.assertEqual(run.dataset_id, 1167)
        self.assertEqual(run.evaluations['f_measure'], 0.624668)
        for i, value in [(0, 0.66233),
                         (1, 0.639286),
                         (2, 0.567143),
                         (3, 0.745833),
                         (4, 0.599638),
                         (5, 0.588801),
                         (6, 0.527976),
                         (7, 0.666365),
                         (8, 0.56759),
                         (9, 0.64621)]:
            self.assertEqual(run.detailed_evaluations['f_measure'][0][i], value)
        assert('weka' in run.tags)
        assert('stacking' in run.tags)

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

        data_content, _, _ = _run_task_get_arffcontent(model, task, class_labels)
        # 2 folds, 5 repeats; keep in mind that this task comes from the test
        # server, the task on the live server is different
        self.assertEqual(len(data_content), 4490)
        for row in data_content:
            # repeat, fold, row_id, 6 confidences, prediction and correct label
            self.assertEqual(len(row), 11)
