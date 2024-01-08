# License: BSD 3-Clause
from __future__ import annotations

import unittest
from distutils.version import LooseVersion

import pytest
import sklearn

from openml.extensions.sklearn import cat, cont
from openml.testing import TestBase


class TestStudyFunctions(TestBase):
    _multiprocess_can_split_ = True
    """Test the example code of Bischl et al. (2018)"""

    @pytest.mark.sklearn()
    @unittest.skipIf(
        LooseVersion(sklearn.__version__) < "0.24",
        reason="columntransformer introduction in 0.24.0",
    )
    def test_Figure1a(self):
        """Test listing in Figure 1a on a single task and the old OpenML100 study.

        The original listing is pasted into the comment below because it the actual unit test
        differs a bit, as for example it does not run for all tasks, but only a single one.

        import openml
        import sklearn.tree, sklearn.preprocessing
        benchmark_suite = openml.study.get_study('OpenML-CC18','tasks') # obtain the benchmark suite
        clf = sklearn.pipeline.Pipeline(steps=[('imputer',sklearn.preprocessing.Imputer()),  ('estimator',sklearn.tree.DecisionTreeClassifier())]) # build a sklearn classifier
        for task_id in benchmark_suite.tasks:                          # iterate over all tasks
            task = openml.tasks.get_task(task_id)                        # download the OpenML task
            X, y = task.get_X_and_y()                                    # get the data (not used in this example)
            openml.config.apikey = 'FILL_IN_OPENML_API_KEY'              # set the OpenML Api Key
            run = openml.runs.run_model_on_task(task,clf)                # run classifier on splits (requires API key)
            score = run.get_metric_fn(sklearn.metrics.accuracy_score) # print accuracy score
            print('Data set: %s; Accuracy: %0.2f' % (task.get_dataset().name,score.mean()))
            run.publish()                                                # publish the experiment on OpenML (optional)
            print('URL for run: %s/run/%d' %(openml.config.server,run.run_id))
        """  # noqa: E501
        import sklearn.metrics
        import sklearn.tree
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        import openml

        benchmark_suite = openml.study.get_study("OpenML100", "tasks")  # obtain the benchmark suite
        cat_imp = OneHotEncoder(handle_unknown="ignore")
        cont_imp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
        clf = Pipeline(
            steps=[("preprocess", ct), ("estimator", sklearn.tree.DecisionTreeClassifier())],
        )  # build a sklearn classifier
        for task_id in benchmark_suite.tasks[:1]:  # iterate over all tasks
            task = openml.tasks.get_task(task_id)  # download the OpenML task
            X, y = task.get_X_and_y()  # get the data (not used in this example)
            openml.config.apikey = openml.config.apikey  # set the OpenML Api Key
            run = openml.runs.run_model_on_task(
                clf,
                task,
                avoid_duplicate_runs=False,
            )  # run classifier on splits (requires API key)
            score = run.get_metric_fn(sklearn.metrics.accuracy_score)  # print accuracy score
            TestBase.logger.info(
                f"Data set: {task.get_dataset().name}; Accuracy: {score.mean():0.2f}",
            )
            run.publish()  # publish the experiment on OpenML (optional)
            TestBase._mark_entity_for_removal("run", run.run_id)
            TestBase.logger.info(
                "collected from {}: {}".format(__file__.split("/")[-1], run.run_id),
            )
            TestBase.logger.info("URL for run: %s/run/%d" % (openml.config.server, run.run_id))
