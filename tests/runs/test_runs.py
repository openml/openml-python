from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import RandomizedSearchCV
import openml
from openml.testing import TestBase
import re
import sys

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock


class TestRun(TestBase):
    def test_run_iris(self):
        task = openml.tasks.get_task(10107)
        clf = LogisticRegression()
        run = openml.runs.run_task(task, clf, seed=15)
        return_code, return_value = run.publish()
        self.assertEqual(return_code, 200)

    def test_publish_trace(self):
        task = openml.tasks.get_task(10107)
        param_distribution = {'C': [10**i for i in range(-5, 5)]}
        clf = LogisticRegression()
        rs = RandomizedSearchCV(clf, param_distribution, n_iter=5)
        run = openml.runs.run_task(task, rs, seed=15)
        return_code, return_value = run.publish()
        self.assertEqual(return_code, 200)

    def test__run_task_get_arffcontent(self):
        task = openml.tasks.get_task(1939)
        class_labels = task.class_labels

        clf = SGDClassifier(loss='hinge', random_state=1)
        self.assertRaisesRegexp(AttributeError, "probability estimates are "
                                                "not available for loss='hinge'",
                                openml.runs.functions._run_task_get_arffcontent,
                                clf, task, class_labels)

        clf = SGDClassifier(loss='log', random_state=1)
        arff_datacontent, arff_tracecontent = openml.runs.functions._run_task_get_arffcontent(
            clf, task, class_labels)
        self.assertIsInstance(arff_datacontent, list)
        # 10 times 10 fold CV of 150 samples
        self.assertEqual(len(arff_datacontent), 1500)
        for arff_line in arff_datacontent:
            self.assertEqual(len(arff_line), 8)
            self.assertGreaterEqual(arff_line[0], 0)
            self.assertLessEqual(arff_line[0], 9)
            self.assertGreaterEqual(arff_line[1], 0)
            self.assertLessEqual(arff_line[1], 9)
            self.assertGreaterEqual(arff_line[2], 0)
            self.assertLessEqual(arff_line[2], 149)
            self.assertAlmostEqual(sum(arff_line[3:6]), 1.0)
            self.assertIn(arff_line[6], ['Iris-setosa', 'Iris-versicolor',
                                         'Iris-virginica'])
            self.assertIn(arff_line[7], ['Iris-setosa', 'Iris-versicolor',
                                         'Iris-virginica'])

        # This assures that the classifier was cloned inside
        # _run_task_get_arffcontent
        self.assertEqual(clf.random_state, 1)

    def test__get_optimization_trajectory(self):
        task = openml.tasks.get_task(3)
        X, y = task.get_X_and_Y()
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        param_distributions = {'n_estimators': [3, 4, 5],
                               'learning_rate': [0.05, 0.1, 0.5],
                               'base_estimator__max_depth': [1, 2]}

        rs = RandomizedSearchCV(estimator=model,
                                param_distributions=param_distributions,
                                n_iter=5, random_state=1)
        rs.fit(X, y)
        header, trace = openml.runs.functions._get_optimization_trajectory(rs,
                                                                           9, 7)
        self.assertEqual(len(trace), 5)
        self.assertEqual(trace[0][0:3], [9, 7, 0])
        self.assertEqual(trace[3][-1], 'true')
        self.assertEqual(len(trace[0]), 8)
        for i, line in enumerate(trace):
            self.assertEqual(line[2], i)

    def test__fix_optimization_trajectory_parameter_names(self):
        input = [['repeat', 'fold', 'iteration',
                  'base_estimator__dummy_component__dummy',
                  'base_estimator__max_depth',
                  'learning_rate', 'n_estimators', 'evaluation', 'selected'],
                 [9, 7, 0, 1, 0.1, 5, 0.67803504380475599, 'false']]
        parameters = [{'oml:name': 'n_estimators'},
                      {'oml:name': 'learning_rate'}]

        dummy_component = mock.Mock(spec=openml.OpenMLFlow)
        dummy_component.parameters = [{'oml:name': 'dummy'}]
        dummy_component.id = 16
        dummy_component.components = []

        dummy_tree_component = mock.Mock(spec=openml.OpenMLFlow)
        dummy_tree_component.parameters = [{'oml:name': 'max_depth'}]
        dummy_tree_component.id = 15
        dummy_tree_component.components = [{'oml:identifier': 'dummy_component',
                                            'oml:flow': dummy_component}]

        components = [{'oml:identifier': 'base_estimator',
                       'oml:flow': dummy_tree_component}]

        traj = openml.runs.functions\
            ._fix_optimization_trajectory_parameter_names(input, 14, parameters,
                                                          components)
        self.assertEqual(traj[0][3:-2], ['16:dummy', '15:max_depth',
                                         '14:learning_rate', '14:n_estimators'])

    def test__create_setup_string(self):
        def strip_version_information(string):
            string = re.sub(r'# Python_[0-9]\.[0-9]\.[0-9]\.\n', '', string)
            string = re.sub(r'# Sklearn_[0-9]\.[0-9][0-9]\.[0-9]\.\n', '', string)
            string = re.sub(r'# NumPy_[0-9]\.[0-9][0-9]\.[0-9]\.\n', '', string)
            string = re.sub(r'# SciPy_[0-9]\.[0-9][0-9]\.[0-9]\.\n', '', string)
            return string

        # Simple test
        self.maxDiff = None
        clf = DecisionTreeClassifier()
        setup_string = openml.runs.run._create_setup_string(clf)
        setup_string = strip_version_information(setup_string)
        self.assertTrue(setup_string.startswith("DecisionTreeClassifier(class_weight=None"))
        clf2 = eval(setup_string)
        self.assertIsInstance(clf2, DecisionTreeClassifier)

        # Test with a classifier nested into an ensemble algorithm
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        setup_string = openml.runs.run._create_setup_string(clf)
        setup_string = strip_version_information(setup_string)
        self.assertTrue(setup_string.startswith("AdaBoostClassifier(algorithm='SAMME.R'"))
        clf2 = eval(setup_string)
        self.assertIsInstance(clf2, AdaBoostClassifier)

        # A more complicated pipeline
        clf = Pipeline([('minmax', MinMaxScaler()),
                        ('fu', FeatureUnion(('minmax1', MinMaxScaler()),
                                            ('minmax2', MinMaxScaler()))),
                        ('classifier', DecisionTreeClassifier())])
        setup_string = openml.runs.run._create_setup_string(clf)
        setup_string = strip_version_information(setup_string)
        self.assertTrue(setup_string.startswith("Pipeline(steps=[('minmax', "))

        clf2 = eval(setup_string)
        self.assertIsInstance(clf2, Pipeline)



