# %% [markdown]
# # Benchmark studies
# How to list, download and upload benchmark studies.
# In contrast to `benchmark suites <https://docs.openml.org/benchmark/#benchmarking-suites>`_ which
# hold a list of tasks, studies hold a list of runs. As runs contain all information on flows and
# tasks, all required information about a study can be retrieved.

# %%
import uuid

from sklearn.ensemble import RandomForestClassifier

import openml


# %% [markdown]
# ##  Listing studies
#
# * Use the output_format parameter to select output type
# * Default gives ``dict``, but we'll use ``dataframe`` to obtain an
#   easier-to-work-with data structure

# %%
studies = openml.study.list_studies(output_format="dataframe", status="all")
print(studies.head(n=10))


# %% [markdown]
# ## Downloading studies

# %% [markdown]
# This is done based on the study ID.

# %%
study = openml.study.get_study(123)
print(study)

# %% [markdown]
# Studies also features a description:

# %%
print(study.description)

# %% [markdown]
# Studies are a container for runs:

# %%
print(study.runs)

# %% [markdown]
# And we can use the evaluation listing functionality to learn more about
# the evaluations available for the conducted runs:

# %%
evaluations = openml.evaluations.list_evaluations(
    function="predictive_accuracy",
    output_format="dataframe",
    study=study.study_id,
)
print(evaluations.head())

# %% [markdown]
# We'll use the test server for the rest of this tutorial.
#
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt

# %%
openml.config.start_using_configuration_for_example()

# %% [markdown]
# ## Uploading studies
#
# Creating a study is as simple as creating any kind of other OpenML entity.
# In this examples we'll create a few runs for the OpenML-100 benchmark
# suite which is available on the OpenML test server.

# %%
# Model to be used
clf = RandomForestClassifier()

# We'll create a study with one run on 3 datasets present in the suite
tasks = [115, 259, 307]

# To verify
suite = openml.study.get_suite(1)
print(all([t_id in suite.tasks for t_id in tasks]))

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


# %%
openml.config.stop_using_configuration_for_example()
# License: BSD 3-Clause
