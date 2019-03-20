import collections
import sys

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing.imputation import Imputer
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import openml
from openml.extensions.sklearn_extension import SklearnExtension
from openml.tasks import TaskTypeEnum
from openml.testing import TestBase


class TestRun(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        super().setUp()
        self.extension = SklearnExtension()

    def test__get_seeded_model(self):
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
            clf_seeded = self.extension.seed_model(clf, const_probe)
            new_params = clf_seeded.get_params()

            randstate_params = [key for key in new_params if
                                key.endswith('random_state')]

            # afterwards, param value is set
            for param in randstate_params:
                self.assertIsInstance(new_params[param], int)
                self.assertIsNotNone(new_params[param])

            if idx == 1:
                self.assertEqual(clf.cv.random_state, 56422)

    def test__get_seeded_model_raises(self):
        # the _set_model_seed_where_none should raise exception if random_state is
        # anything else than an int
        randomized_clfs = [
            BaggingClassifier(random_state=np.random.RandomState(42)),
            DummyClassifier(random_state="OpenMLIsGreat")
        ]

        for clf in randomized_clfs:
            with self.assertRaises(ValueError):
                self.extension.seed_model(model=clf, seed=42)

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
            arff_line = self.extension._prediction_to_row(
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

    def test__run_model_on_fold(self):
        task = openml.tasks.get_task(7)
        num_instances = 320
        num_folds = 1
        num_repeats = 1

        clf = SGDClassifier(loss='log', random_state=1)
        can_measure_runtime = sys.version_info[:2] >= (3, 3)
        res = self.extension.run_model_on_fold(
            clf, task, 0, 0, 0, can_measure_runtime=can_measure_runtime,
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

    def _check_fold_evaluations(self, fold_evaluations, num_repeats, num_folds,
                                max_time_allowed=60000,
                                task_type=(TaskTypeEnum.
                                           SUPERVISED_CLASSIFICATION)):
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
            'usercpu_time_millis_testing': (0, max_time_allowed),
            'usercpu_time_millis_training': (0, max_time_allowed),
            # should take at least one millisecond (?)
            'usercpu_time_millis': (0, max_time_allowed)}

        if task_type == TaskTypeEnum.SUPERVISED_CLASSIFICATION or \
                task_type == TaskTypeEnum.LEARNING_CURVE:
            check_measures['predictive_accuracy'] = (0, 1)
        elif task_type == TaskTypeEnum.SUPERVISED_REGRESSION:
            check_measures['mean_absolute_error'] = (0, float("inf"))

        self.assertIsInstance(fold_evaluations, dict)
        if sys.version_info[:2] >= (3, 3):
            # this only holds if we are allowed to record time (otherwise some
            # are missing)
            self.assertEqual(set(fold_evaluations.keys()),
                             set(check_measures.keys()))

        for measure in check_measures.keys():
            if measure in fold_evaluations:
                num_rep_entrees = len(fold_evaluations[measure])
                self.assertEqual(num_rep_entrees, num_repeats)
                min_val = check_measures[measure][0]
                max_val = check_measures[measure][1]
                for rep in range(num_rep_entrees):
                    num_fold_entrees = len(fold_evaluations[measure][rep])
                    self.assertEqual(num_fold_entrees, num_folds)
                    for fold in range(num_fold_entrees):
                        evaluation = fold_evaluations[measure][rep][fold]
                        self.assertIsInstance(evaluation, float)
                        self.assertGreaterEqual(evaluation, min_val)
                        self.assertLessEqual(evaluation, max_val)
