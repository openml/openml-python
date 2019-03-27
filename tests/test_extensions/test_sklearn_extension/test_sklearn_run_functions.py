import collections
import json
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.imputation import Imputer

import openml
from openml.extensions.sklearn.run_functions import (
    _extract_trace_data,
    _prediction_to_row,
    seed_model,
    run_model_on_fold,
    obtain_arff_trace,
)

from openml.testing import TestBase
from openml.runs.trace import OpenMLRunTrace


class TestRun(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp(n_levels=2)

    ################################################################################################
    # Test methods for performing runs with this extension module

    def test_seed_model(self):
        # randomized models that are initialized without seeds, can be seeded
        randomized_clfs = [
            BaggingClassifier(),
            RandomizedSearchCV(RandomForestClassifier(),
                               {"max_depth": [3, None],
                                "max_features": [1, 2, 3, 4],
                                "bootstrap": [True, False],
                                "criterion": ["gini", "entropy"],
                                "random_state": [-1, 0, 1, 2]},
                               cv=StratifiedKFold(n_splits=2, shuffle=True)),
            DummyClassifier()
        ]

        for idx, clf in enumerate(randomized_clfs):
            const_probe = 42
            all_params = clf.get_params()
            params = [key for key in all_params if
                      key.endswith('random_state')]
            self.assertGreater(len(params), 0)

            # before param value is None
            for param in params:
                self.assertIsNone(all_params[param])

            # now seed the params
            clf_seeded = seed_model(clf, const_probe)
            new_params = clf_seeded.get_params()

            randstate_params = [key for key in new_params if
                                key.endswith('random_state')]

            # afterwards, param value is set
            for param in randstate_params:
                self.assertIsInstance(new_params[param], int)
                self.assertIsNotNone(new_params[param])

            if idx == 1:
                self.assertEqual(clf.cv.random_state, 56422)

    def test_seed_model_raises(self):
        # the _set_model_seed_where_none should raise exception if random_state is
        # anything else than an int
        randomized_clfs = [
            BaggingClassifier(random_state=np.random.RandomState(42)),
            DummyClassifier(random_state="OpenMLIsGreat")
        ]

        for clf in randomized_clfs:
            with self.assertRaises(ValueError):
                seed_model(model=clf, seed=42)

    def test_run_model_on_fold(self):
        task = openml.tasks.get_task(7)
        num_instances = 320
        num_folds = 1
        num_repeats = 1

        clf = SGDClassifier(loss='log', random_state=1)
        # TODO add some mocking here to actually test the innards of this function, too!
        res = run_model_on_fold(
            clf, task, 0, 0, 0,
            add_local_measures=True)

        arff_datacontent, arff_tracecontent, user_defined_measures, model = res
        # predictions
        self.assertIsInstance(arff_datacontent, list)
        # trace. SGD does not produce any
        self.assertIsInstance(arff_tracecontent, list)
        self.assertEqual(len(arff_tracecontent), 0)

        fold_evaluations = collections.defaultdict(
            lambda: collections.defaultdict(dict))
        for measure in user_defined_measures:
            fold_evaluations[measure][0][0] = user_defined_measures[measure]

        self._check_fold_evaluations(fold_evaluations, num_repeats, num_folds,
                                     task_type=task.task_type_id)

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

    def test__prediction_to_row(self):
        repeat_nr = 0
        fold_nr = 0
        clf = Pipeline(steps=[
            ('Imputer', Imputer(strategy='mean')),
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
        sample_nr = 0  # default for this task
        for idx in range(0, len(test_X)):
            arff_line = _prediction_to_row(
                rep_no=repeat_nr,
                fold_no=fold_nr,
                sample_no=sample_nr,
                row_id=idx,
                correct_label=task.class_labels[test_y[idx]],
                predicted_label=predY[idx],
                predicted_probabilities=probaY[idx],
                class_labels=task.class_labels,
                model_classes_mapping=clf.classes_,
            )

            self.assertIsInstance(arff_line, list)
            self.assertEqual(len(arff_line), 6 + len(task.class_labels))
            self.assertEqual(arff_line[0], repeat_nr)
            self.assertEqual(arff_line[1], fold_nr)
            self.assertEqual(arff_line[2], sample_nr)
            self.assertEqual(arff_line[3], idx)
            sum_ = 0.0
            for att_idx in range(4, 4 + len(task.class_labels)):
                self.assertIsInstance(arff_line[att_idx], float)
                self.assertGreaterEqual(arff_line[att_idx], 0.0)
                self.assertLessEqual(arff_line[att_idx], 1.0)
                sum_ += arff_line[att_idx]
            self.assertAlmostEqual(sum_, 1.0)

            self.assertIn(arff_line[-1], task.class_labels)
            self.assertIn(arff_line[-2], task.class_labels)
        pass

    def test__extract_trace_data(self):

        param_grid = {"hidden_layer_sizes": [[5, 5], [10, 10], [20, 20]],
                      "activation": ['identity', 'logistic', 'tanh', 'relu'],
                      "learning_rate_init": [0.1, 0.01, 0.001, 0.0001],
                      "max_iter": [10, 20, 40, 80]}
        num_iters = 10
        task = openml.tasks.get_task(20)
        clf = RandomizedSearchCV(MLPClassifier(), param_grid, num_iters)
        # just run the task
        train, _ = task.get_train_test_split_indices(0, 0)
        X, y = task.get_X_and_y()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(X[train], y[train])

        # check num layers of MLP
        self.assertIn(clf.best_estimator_.hidden_layer_sizes, param_grid['hidden_layer_sizes'])

        trace_list = _extract_trace_data(clf, rep_no=0, fold_no=0)
        trace = obtain_arff_trace(clf, trace_list)

        self.assertIsInstance(trace, OpenMLRunTrace)
        self.assertIsInstance(trace_list, list)
        self.assertEqual(len(trace_list), num_iters)

        for trace_iteration in iter(trace):
            self.assertEqual(trace_iteration.repeat, 0)
            self.assertEqual(trace_iteration.fold, 0)
            self.assertGreaterEqual(trace_iteration.iteration, 0)
            self.assertLessEqual(trace_iteration.iteration, num_iters)
            self.assertIsNone(trace_iteration.setup_string)
            self.assertIsInstance(trace_iteration.evaluation, float)
            self.assertTrue(np.isfinite(trace_iteration.evaluation))
            self.assertIsInstance(trace_iteration.selected, bool)

            self.assertEqual(len(trace_iteration.parameters), len(param_grid))
            for param in param_grid:

                # Prepend with the "parameter_" prefix
                param_in_trace = "parameter_%s" % param
                self.assertIn(param_in_trace, trace_iteration.parameters)
                param_value = json.loads(trace_iteration.parameters[param_in_trace])
                self.assertTrue(param_value in param_grid[param])
