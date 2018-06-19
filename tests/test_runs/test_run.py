import numpy as np
import random
import os
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from openml.testing import TestBase
from openml.flows.sklearn_converter import sklearn_to_flow
from openml import OpenMLRun
import openml


class TestRun(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take
    # less than 1 seconds

    def test_parse_parameters_flow_not_on_server(self):

        model = LogisticRegression()
        flow = sklearn_to_flow(model)
        self.assertRaisesRegexp(
            ValueError, 'Flow sklearn.linear_model.logistic.LogisticRegression'
            ' has no flow_id!', OpenMLRun._parse_parameters, flow)

        model = AdaBoostClassifier(base_estimator=LogisticRegression())
        flow = sklearn_to_flow(model)
        flow.flow_id = 1
        self.assertRaisesRegexp(
            ValueError, 'Flow sklearn.linear_model.logistic.LogisticRegression'
            ' has no flow_id!', OpenMLRun._parse_parameters, flow)

    def test_parse_parameters(self):

        model = RandomizedSearchCV(
            estimator=RandomForestClassifier(n_estimators=5),
            param_distributions={
                "max_depth": [3, None],
                "max_features": [1, 2, 3, 4],
                "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "bootstrap": [True, False], "criterion": ["gini", "entropy"]},
            cv=StratifiedKFold(n_splits=2, random_state=1),
            n_iter=5)
        flow = sklearn_to_flow(model)
        flow.flow_id = 1
        flow.components['estimator'].flow_id = 2
        parameters = OpenMLRun._parse_parameters(flow)
        for parameter in parameters:
            self.assertIsNotNone(parameter['oml:component'], msg=parameter)
            if parameter['oml:name'] == 'n_estimators':
                self.assertEqual(parameter['oml:value'], '5')
                self.assertEqual(parameter['oml:component'], 2)

    def test_tagging(self):

        runs = openml.runs.list_runs(size=1)
        run_id = list(runs.keys())[0]
        run = openml.runs.get_run(run_id)
        tag = "testing_tag_{}_{}".format(self.id(), time())
        run_list = openml.runs.list_runs(tag=tag)
        self.assertEqual(len(run_list), 0)
        run.push_tag(tag)
        run_list = openml.runs.list_runs(tag=tag)
        self.assertEqual(len(run_list), 1)
        self.assertIn(run_id, run_list)
        run.remove_tag(tag)
        run_list = openml.runs.list_runs(tag=tag)
        self.assertEqual(len(run_list), 0)

    def _test_run_obj_equals(self, run, run_prime):
        for dictionary in ['evaluations', 'fold_evaluations', 'sample_evaluations']:
            if getattr(run, dictionary) is not None:
                self.assertDictEqual(getattr(run, dictionary), getattr(run_prime, dictionary))
            else:
                # should be none or empty
                other = getattr(run_prime, dictionary)
                if other is not None:
                    self.assertDictEqual(other, dict())
        self.assertEqual(run._create_description_xml(), run_prime._create_description_xml())

        numeric_part = np.array(np.array(run.data_content)[:, 0:-2], dtype=float)
        numeric_part_prime = np.array(np.array(run_prime.data_content)[:, 0:-2], dtype=float)
        string_part = np.array(run.data_content)[:, -2:]
        string_part_prime = np.array(run_prime.data_content)[:, -2:]
        # JvR: Python 2.7 requires an almost equal check, rather than an equals check
        np.testing.assert_array_almost_equal(numeric_part, numeric_part_prime)
        np.testing.assert_array_equal(string_part, string_part_prime)

        if run.trace_content is not None:
            def _check_array(array, type_):
                for line in array:
                    for entry in line:
                        self.assertIsInstance(entry, type_)

            int_part = [line[:3] for line in run.trace_content]
            _check_array(int_part, int)
            int_part_prime = [line[:3] for line in run_prime.trace_content]
            _check_array(int_part_prime, int)

            float_part = np.array(np.array(run.trace_content)[:, 3:4], dtype=float)
            float_part_prime = np.array(np.array(run_prime.trace_content)[:, 3:4], dtype=float)
            bool_part = [line[4] for line in run.trace_content]
            bool_part_prime = [line[4] for line in run_prime.trace_content]
            for bp, bpp in zip(bool_part, bool_part_prime):
                self.assertIn(bp, ['true', 'false'])
                self.assertIn(bpp, ['true', 'false'])
            string_part = np.array(run.trace_content)[:, 5:]
            string_part_prime = np.array(run_prime.trace_content)[:, 5:]
            # JvR: Python 2.7 requires an almost equal check, rather than an equals check
            np.testing.assert_array_almost_equal(int_part, int_part_prime)
            np.testing.assert_array_almost_equal(float_part, float_part_prime)
            self.assertEqual(bool_part, bool_part_prime)
            np.testing.assert_array_equal(string_part, string_part_prime)
        else:
            self.assertIsNone(run_prime.trace_content)

    def test_to_from_filesystem_vanilla(self):
        model = Pipeline([
            ('imputer', Imputer(strategy='mean')),
            ('classifier', DecisionTreeClassifier(max_depth=1)),
        ])
        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(task, model, add_local_measures=False)

        cache_path = os.path.join(self.workdir, 'runs', str(random.getrandbits(128)))
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        self._test_run_obj_equals(run, run_prime)
        run_prime.publish()

    def test_to_from_filesystem_search(self):
        model = Pipeline([
            ('imputer', Imputer(strategy='mean')),
            ('classifier', DecisionTreeClassifier(max_depth=1)),
        ])
        model = GridSearchCV(
            estimator=model,
            param_grid={
                "classifier__max_depth": [1, 2, 3, 4, 5],
                "imputer__strategy": ['mean', 'median'],
            }
        )

        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(task, model, add_local_measures=False)

        cache_path = os.path.join(self.workdir, 'runs', str(random.getrandbits(128)))
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        self._test_run_obj_equals(run, run_prime)
        run_prime.publish()
