"""
=================
Benchmark studies
=================

How to list, download and upload benchmark studies.

In contrast to `benchmark suites <https://docs.openml.org/benchmark/#benchmarking-suites>`_ which
hold a list of tasks, studies hold a list of runs. As runs contain all information on flows and
tasks, all required information about a study can be retrieved.
"""
############################################################################

# License: BSD 3-Clause

import uuid

import numpy as np
import sklearn.tree
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

import openml


############################################################################
# .. warning:: This example uploads data. For that reason, this example
#   connects to the test server at test.openml.org before doing so.
#   This prevents the crowding of the main server with example datasets,
#   tasks, runs, and so on.
############################################################################


############################################################################
# Listing studies
# ***************
#
# * Use the output_format parameter to select output type
# * Default gives ``dict``, but we'll use ``dataframe`` to obtain an
#   easier-to-work-with data structure

studies = openml.study.list_studies(output_format="dataframe", status="all")
print(studies.head(n=10))


############################################################################
# Downloading studies
# ===================

############################################################################
# This is done based on the study ID.
study = openml.study.get_study(123)
print(study)

############################################################################
# Studies also features a description:
print(study.description)

############################################################################
# Studies are a container for runs:
print(study.runs)

############################################################################
# And we can use the evaluation listing functionality to learn more about
# the evaluations available for the conducted runs:
evaluations = openml.evaluations.list_evaluations(
    function="predictive_accuracy", output_format="dataframe", study=study.study_id,
)
print(evaluations.head())

###########################################################from openml.testing import cat, cont#################
# Uploading studies
# =================
#
# Creating a study is as simple as creating any kind of other OpenML entity.
# In this examples we'll create a few runs for the OpenML-100 benchmark
# suite which is available on the OpenML test server.

openml.config.start_using_configuration_for_example()

# Model that can handle missing values
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


# Helper functions to return required columns for ColumnTransformer
def cont(X):
    return X.dtypes != "category"


def cat(X):
    return X.dtypes == "category"


cat_imp = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse=False),
    TruncatedSVD(),
)
ct = ColumnTransformer(
    [("cat", cat_imp, cat), ("cont", FunctionTransformer(lambda x: x, validate=False), cont)]
)
clf = sklearn.pipeline.Pipeline(
    steps=[("transform", ct), ("estimator", HistGradientBoostingClassifier()),]
)

suite = openml.study.get_suite(1)
# We'll create a study with one run on three random datasets each
tasks = np.random.choice(suite.tasks, size=3, replace=False)
run_ids = []
for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    run = openml.runs.run_model_on_task(clf, task)
    run.publish()
    run_ids.append(run.run_id)

# The study needs a machine-readable and unique alias. To obtain this,
# we simply generate a random uuid.
alias = uuid.uuid4().hex

new_study = openml.study.create_study(
    name="Test-Study",
    description="Test study for the Python tutorial on studies",
    run_ids=run_ids,
    alias=alias,
    benchmark_suite=suite.study_id,
)
new_study.publish()
print(new_study)


############################################################################
openml.config.stop_using_configuration_for_example()
