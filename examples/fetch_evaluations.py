"""
Tasks
=====

A tutorial on how to fetch evalutions on a task.
"""

import openml
# import pandas as pd
from pprint import pprint

############################################################################
#
# Evalutions contain details of all runs and the resulting results that
# was uploaded for those settings - data, flow, task, etc.
# The listing functions take optional parameters which can be used to filter
# results and fetch only the evaluations required.
#
# In this example, we'll primarily see how to retrieve the results for a
# particular task and attempt to compare performance of different runs.

############################################################################
# Listing evaluations
# ^^^^^^^^^^^^^^^^^^^
#
# We shall retrieve a small list and test the listing function for evaluations
evals = openml.evaluations.list_evaluations(function='predictive_accuracy', size=10)
pprint(evals)
# To have a tabular output
openml.evaluations.list_evaluations(function='predictive_accuracy', size=10,
                                    output_format='dataframe')
# Using other evaluation metrics
openml.evaluations.list_evaluations(function='precision', size=10,
                                    output_format='dataframe')

# Listing tasks
# ^^^^^^^^^^^^^
#
# We will start by displaying a simple *supervised classification* task:
task_id = 167140        # https://www.openml.org/t/167140
task = openml.tasks.get_tasks([task_id])[0]
pprint(vars(task))

# Obtaining all the evaluations for the task
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
evals = openml.evaluations.list_evaluations(function='predictive_accuracy', task=[task_id],
                                            output_format='dataframe')
# Displaying the first 10 rows
pprint(evals.head(n=10))
# Sorting the evaluations in decreasing order of the metric chosen
evals = evals.sort_values(by='value', ascending=False)
pprint(evals.head())

# Obtain CDF
# ^^^^^^^^^^
#
from matplotlib import pyplot as plt


def plot_cdf(values, metric='predictive_accuracy'):
    plt.hist(values, density=True, histtype='step', cumulative=True)
    plt.xlim(max(0, min(values) - 0.1), 1)
    plt.title('CDF')
    plt.xlabel(metric)
    plt.ylabel('Likelihood')
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.show()


plot_cdf(evals.value)
