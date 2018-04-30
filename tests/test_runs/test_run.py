import numpy as np
import random
import os
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

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

        numeric_part = np.array(run.data_content)[:, 0:-2]
        numeric_part_prime = np.array(run_prime.data_content)[:, 0:-2]
        string_part = np.array(run.data_content)[:, -2:]
        string_part_prime = np.array(run_prime.data_content)[:, -2:]
        np.testing.assert_array_equal(np.array(numeric_part, dtype=float), np.array(numeric_part_prime, dtype=float))
        np.testing.assert_array_equal(np.array(string_part), np.array(string_part_prime))

        if run.trace_content is not None:
            numeric_part = np.array(run.trace_content)[:, 0:-2]
            numeric_part_prime = np.array(run_prime.trace_content)[:, 0:-2]
            string_part = np.array(run.trace_content)[:, -2:]
            string_part_prime = np.array(run_prime.trace_content)[:, -2:]
            np.testing.assert_array_equal(np.array(numeric_part, dtype=float),
                                          np.array(numeric_part_prime, dtype=float))
            np.testing.assert_array_equal(np.array(string_part), np.array(string_part_prime))

    def test_to_from_filesystem_vanilla(self):
        model = DecisionTreeClassifier(max_depth=1)
        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(task, model)

        cache_path = os.path.join(self.workdir, 'runs', str(random.getrandbits(128)))
        os.makedirs(cache_path)
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        self._test_run_obj_equals(run, run_prime)

    def test_to_from_filesystem_search(self):
        model = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={"max_depth": [1, 2, 3, 4, 5]})

        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(task, model)

        cache_path = os.path.join(self.workdir, 'runs', str(random.getrandbits(128)))
        os.makedirs(cache_path)
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        self._test_run_obj_equals(run, run_prime)