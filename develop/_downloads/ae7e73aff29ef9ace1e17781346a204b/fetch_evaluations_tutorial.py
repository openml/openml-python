"""
====================
Fetching Evaluations
====================

Evaluations contain a concise summary of the results of all runs made. Each evaluation
provides information on the dataset used, the flow applied, the setup used, the metric
evaluated, and the result obtained on the metric, for each such run made. These collection
of results can be used for efficient benchmarking of an algorithm and also allow transparent
reuse of results from previous experiments on similar parameters.

In this example, we shall do the following:

* Retrieve evaluations based on different metrics
* Fetch evaluations pertaining to a specific task
* Sort the obtained results in descending order of the metric
* Plot a cumulative distribution function for the evaluations
* Compare the top 10 performing flows based on the evaluation performance
* Retrieve evaluations with hyperparameter settings
"""

############################################################################

# License: BSD 3-Clause

import openml

############################################################################
# Listing evaluations
# *******************
# Evaluations can be retrieved from the database in the chosen output format.
# Required filters can be applied to retrieve results from runs as required.

# We shall retrieve a small set (only 10 entries) to test the listing function for evaluations
openml.evaluations.list_evaluations(
    function="predictive_accuracy", size=10, output_format="dataframe"
)

# Using other evaluation metrics, 'precision' in this case
evals = openml.evaluations.list_evaluations(
    function="precision", size=10, output_format="dataframe"
)

# Querying the returned results for precision above 0.98
print(evals[evals.value > 0.98])

#############################################################################
# Viewing a sample task
# =====================
# Over here we shall briefly take a look at the details of the task.

# We will start by displaying a simple *supervised classification* task:
task_id = 167140  # https://www.openml.org/t/167140
task = openml.tasks.get_task(task_id)
print(task)

#############################################################################
# Obtaining all the evaluations for the task
# ==========================================
# We'll now obtain all the evaluations that were uploaded for the task
# we displayed previously.
# Note that we now filter the evaluations based on another parameter 'task'.

metric = "predictive_accuracy"
evals = openml.evaluations.list_evaluations(
    function=metric, tasks=[task_id], output_format="dataframe"
)
# Displaying the first 10 rows
print(evals.head(n=10))
# Sorting the evaluations in decreasing order of the metric chosen
evals = evals.sort_values(by="value", ascending=False)
print("\nDisplaying head of sorted dataframe: ")
print(evals.head())

#############################################################################
# Obtaining CDF of metric for chosen task
# ***************************************
# We shall now analyse how the performance of various flows have been on this task,
# by seeing the likelihood of the accuracy obtained across all runs.
# We shall now plot a cumulative distributive function (CDF) for the accuracies obtained.

from matplotlib import pyplot as plt


def plot_cdf(values, metric="predictive_accuracy"):
    max_val = max(values)
    n, bins, patches = plt.hist(values, density=True, histtype="step", cumulative=True, linewidth=3)
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.xlim(max(0, min(values) - 0.1), 1)
    plt.title("CDF")
    plt.xlabel(metric)
    plt.ylabel("Likelihood")
    plt.grid(visible=True, which="major", linestyle="-")
    plt.minorticks_on()
    plt.grid(visible=True, which="minor", linestyle="--")
    plt.axvline(max_val, linestyle="--", color="gray")
    plt.text(max_val, 0, "%.3f" % max_val, fontsize=9)
    plt.show()


plot_cdf(evals.value, metric)
# This CDF plot shows that for the given task, based on the results of the
# runs uploaded, it is almost certain to achieve an accuracy above 52%, i.e.,
# with non-zero probability. While the maximum accuracy seen till now is 96.5%.

#############################################################################
# Comparing top 10 performing flows
# *********************************
# Let us now try to see which flows generally performed the best for this task.
# For this, we shall compare the top performing flows.

import numpy as np
import pandas as pd


def plot_flow_compare(evaluations, top_n=10, metric="predictive_accuracy"):
    # Collecting the top 10 performing unique flow_id
    flow_ids = evaluations.flow_id.unique()[:top_n]

    df = pd.DataFrame()
    # Creating a data frame containing only the metric values of the selected flows
    #   assuming evaluations is sorted in decreasing order of metric
    for i in range(len(flow_ids)):
        flow_values = evaluations[evaluations.flow_id == flow_ids[i]].value
        df = pd.concat([df, flow_values], ignore_index=True, axis=1)
    fig, axs = plt.subplots()
    df.boxplot()
    axs.set_title("Boxplot comparing " + metric + " for different flows")
    axs.set_ylabel(metric)
    axs.set_xlabel("Flow ID")
    axs.set_xticklabels(flow_ids)
    axs.grid(which="major", linestyle="-", linewidth="0.5", color="gray", axis="y")
    axs.minorticks_on()
    axs.grid(which="minor", linestyle="--", linewidth="0.5", color="gray", axis="y")
    # Counting the number of entries for each flow in the data frame
    #   which gives the number of runs for each flow
    flow_freq = list(df.count(axis=0, numeric_only=True))
    for i in range(len(flow_ids)):
        axs.text(i + 1.05, np.nanmin(df.values), str(flow_freq[i]) + "\nrun(s)", fontsize=7)
    plt.show()


plot_flow_compare(evals, metric=metric, top_n=10)
# The boxplots below show how the flows perform across multiple runs on the chosen
# task. The green horizontal lines represent the median accuracy of all the runs for
# that flow (number of runs denoted at the bottom of the boxplots). The higher the
# green line, the better the flow is for the task at hand. The ordering of the flows
# are in the descending order of the higest accuracy value seen under that flow.

# Printing the corresponding flow names for the top 10 performing flow IDs
top_n = 10
flow_ids = evals.flow_id.unique()[:top_n]
flow_names = evals.flow_name.unique()[:top_n]
for i in range(top_n):
    print((flow_ids[i], flow_names[i]))

#############################################################################
# Obtaining evaluations with hyperparameter settings
# ==================================================
# We'll now obtain the evaluations of a task and a flow with the hyperparameters

# List evaluations in descending order based on predictive_accuracy with
# hyperparameters
evals_setups = openml.evaluations.list_evaluations_setups(
    function="predictive_accuracy", tasks=[31], size=100, sort_order="desc"
)

""
print(evals_setups.head())

""
# Return evaluations for flow_id in descending order based on predictive_accuracy
# with hyperparameters. parameters_in_separate_columns returns parameters in
# separate columns
evals_setups = openml.evaluations.list_evaluations_setups(
    function="predictive_accuracy", flows=[6767], size=100, parameters_in_separate_columns=True
)

""
print(evals_setups.head(10))

""
