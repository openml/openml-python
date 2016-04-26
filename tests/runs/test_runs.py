from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
import openml
from openml.testing import TestBase
import re


class TestRun(TestBase):
    def test_run_iris(self):
        task = openml.tasks.get_task(10107)
        clf = LogisticRegression()
        run = openml.runs.run_task(task, clf)
        return_code, return_value = run.publish()
        self.assertEqual(return_code, 200)
        # self.assertTrue("This is a read-only account" in return_value)

    def test__run_task_get_arffcontent(self):
        task = openml.tasks.get_task(1939)
        class_labels = task.class_labels

        clf = SGDClassifier(loss='hinge', random_state=1)
        self.assertRaisesRegex(AttributeError, "probability estimates are not available for loss='hinge'",
                               openml.runs.functions._run_task_get_arffcontent,
                               clf, task, class_labels)

        clf = SGDClassifier(loss='log', random_state=1)
        arff_datacontent = openml.runs.functions._run_task_get_arffcontent(
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

    def test__create_setup_string(self):
        def strip_version_information(string):
            string = re.sub(r'# Python_[0-9]\.[0-9]\.[0-9]\.\n', '', string)
            string = re.sub(r'# Sklearn_[0-9]\.[0-9][0-9]\.[0-9]\.\n', '', string)
            string = re.sub(r'# NumPy_[0-9]\.[0-9][0-9]\.[0-9]\.\n', '', string)
            string = re.sub(r'# SciPy_[0-9]\.[0-9][0-9]\.[0-9]\.\n', '', string)
            return string

        # Simple test
        self.maxDiff = None
        clf = LogisticRegression()
        setup_string = openml.runs.run._create_setup_string(clf)
        setup_string = strip_version_information(setup_string)
        self.assertEqual(setup_string,
                         "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n"
                         "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n"
                         "          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,\n"
                         "          verbose=0, warm_start=False)")
        clf2 = eval(setup_string)
        self.assertIsInstance(clf2, LogisticRegression)

        # Test with a classifier nested into an ensemble algorithm
        clf = AdaBoostClassifier(base_estimator=LogisticRegression())
        setup_string = openml.runs.run._create_setup_string(clf)
        setup_string = strip_version_information(setup_string)
        self.assertEqual(setup_string,
                         "AdaBoostClassifier(algorithm='SAMME.R',\n"
                         "          base_estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n"
                         "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n"
                         "          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,\n"
                         "          verbose=0, warm_start=False),\n"
                         "          learning_rate=1.0, n_estimators=50, random_state=None)")
        clf2 = eval(setup_string)
        self.assertIsInstance(clf2, AdaBoostClassifier)

        # A more complicated pipeline
        clf = Pipeline([('minmax', MinMaxScaler()),
                        ('fu', FeatureUnion(('minmax1', MinMaxScaler()),
                                            ('minmax2', MinMaxScaler()))),
                        ('classifier', LogisticRegression())])
        setup_string = openml.runs.run._create_setup_string(clf)
        setup_string = strip_version_information(setup_string)
        self.assertEqual(setup_string, "Pipeline(steps=[('minmax', MinMaxScaler(copy=True, feature_range=(0, 1))), ('fu', FeatureUnion(n_jobs=('minmax2', MinMaxScaler(copy=True, feature_range=(0, 1))),\n"
                                       "       transformer_list=('minmax1', MinMaxScaler(copy=True, feature_range=(0, 1))),\n"
                                       "       transformer_weights=None)), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n"
                                       "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n"
                                       "          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,\n"
                                       "          verbose=0, warm_start=False))])")

        clf2 = eval(setup_string)
        self.assertIsInstance(clf2, Pipeline)



