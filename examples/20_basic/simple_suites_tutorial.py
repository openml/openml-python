"""
================
Benchmark suites
================

This is a brief showcase of OpenML benchmark suites, which were introduced by
`Bischl et al. (2019) <https://arxiv.org/abs/1708.03731v2>`_. Benchmark suites standardize the
datasets and splits to be used in an experiment or paper. They are fully integrated into OpenML
and simplify both the sharing of the setup and the results.
"""

# License: BSD 3-Clause

import openml

####################################################################################################
# OpenML-CC18
# ===========
#
# As an example we have a look at the OpenML-CC18, which is a suite of 72 classification datasets
# from OpenML which were carefully selected to be usable by many algorithms and also represent
# datasets commonly used in machine learning research. These are all datasets from mid-2018 that
# satisfy a large set of clear requirements for thorough yet practical benchmarking:
#
# 1. the number of observations are between 500 and 100,000 to focus on medium-sized datasets,
# 2. the number of features does not exceed 5,000 features to keep the runtime of the algorithms
#    low
# 3. the target attribute has at least two classes with no class having less than 20 observations
# 4. the ratio of the minority class and the majority class is above 0.05 (to eliminate highly
#    imbalanced datasets which require special treatment for both algorithms and evaluation
#    measures).
#
# A full description can be found in the `OpenML benchmarking docs
# <https://docs.openml.org/benchmark/#openml-cc18>`_.
#
# In this example we'll focus on how to use benchmark suites in practice.

####################################################################################################
# Downloading benchmark suites
# ============================

suite = openml.study.get_suite(99)
print(suite)

####################################################################################################
# The benchmark suite does not download the included tasks and datasets itself, but only contains
# a list of which tasks constitute the study.
#
# Tasks can then be accessed via

tasks = suite.tasks
print(tasks)

####################################################################################################
# and iterated over for benchmarking. For speed reasons we only iterate over the first three tasks:

for task_id in tasks[:3]:
    task = openml.tasks.get_task(task_id)
    print(task)

####################################################################################################
# Further examples
# ================
#
# * `Advanced benchmarking suites tutorial <../30_extended/suites_tutorial.html>`_
# * `Benchmarking studies tutorial <../30_extended/study_tutorial.html>`_
# * `Using studies to compare linear and non-linear classifiers
#   <../40_paper/2018_ida_strang_example.html>`_
