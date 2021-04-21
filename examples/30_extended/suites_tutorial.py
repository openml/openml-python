"""
================
Benchmark suites
================

How to list, download and upload benchmark suites.

If you want to learn more about benchmark suites, check out our
brief introductory tutorial :ref:`sphx_glr_examples_20_basic_simple_suites_tutorial.py` or the
`OpenML benchmark docs <https://docs.openml.org/benchmark/#benchmarking-suites>`_.
"""
############################################################################

# License: BSD 3-Clause

import uuid

import numpy as np

import openml

############################################################################
# .. warning:: This example uploads data. For that reason, this example
#   connects to the test server at test.openml.org before doing so.
#   This prevents the main server from crowding with example datasets,
#   tasks, runs, and so on.
#
############################################################################


############################################################################
# Listing suites
# **************
#
# * Use the output_format parameter to select output type
# * Default gives ``dict``, but we'll use ``dataframe`` to obtain an
#   easier-to-work-with data structure

suites = openml.study.list_suites(output_format="dataframe", status="all")
print(suites.head(n=10))

############################################################################
# Downloading suites
# ==================

############################################################################
# This is done based on the dataset ID.
suite = openml.study.get_suite(99)
print(suite)

############################################################################
# Suites also feature a description:
print(suite.description)

############################################################################
# Suites are a container for tasks:
print(suite.tasks)

############################################################################
# And we can use the task listing functionality to learn more about them:
tasks = openml.tasks.list_tasks(output_format="dataframe")

# Using ``@`` in `pd.DataFrame.query <
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_
# accesses variables outside of the current dataframe.
tasks = tasks.query("tid in @suite.tasks")
print(tasks.describe().transpose())

############################################################################
# Uploading suites
# ================
#
# Uploading suites is as simple as uploading any kind of other OpenML
# entity - the only reason why we need so much code in this example is
# because we upload some random data.
#
# .. warning:: This example uploads data. For that reason, this example
#   connects to the test server at test.openml.org. This prevents the main
#   server from crowding with example datasets, tasks, runs, and so on. The
#   use of this test server can affect behaviour and performance of the
#   OpenML-Python API.

openml.config.start_using_configuration_for_example()

# We'll take a random subset of at least ten tasks of all available tasks on
# the test server:
all_tasks = list(openml.tasks.list_tasks().keys())
task_ids_for_suite = sorted(np.random.choice(all_tasks, replace=False, size=20))

# The study needs a machine-readable and unique alias. To obtain this,
# we simply generate a random uuid.

alias = uuid.uuid4().hex

new_suite = openml.study.create_benchmark_suite(
    name="Test-Suite",
    description="Test suite for the Python tutorial on benchmark suites",
    task_ids=task_ids_for_suite,
    alias=alias,
)
new_suite.publish()
print(new_suite)


############################################################################
openml.config.stop_using_configuration_for_example()
