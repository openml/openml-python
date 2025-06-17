"""
================================
Plotting hyperparameter surfaces
================================
"""

# License: BSD 3-Clause

import numpy as np

import openml

####################################################################################################
# First step - obtaining the data
# ===============================
# First, we need to choose an SVM flow, for example 8353, and a task. Finding the IDs of them are
# not part of this tutorial, this could for example be done via the website.
#
# For this we use the function ``list_evaluations_setup`` which can automatically join
# evaluations conducted by the server with the hyperparameter settings extracted from the
# uploaded runs (called *setup*).
df = openml.evaluations.list_evaluations_setups(
    function="predictive_accuracy",
    flows=[8353],
    tasks=[6],
    # Using this flag incorporates the hyperparameters into the returned dataframe. Otherwise,
    # the dataframe would contain a field ``paramaters`` containing an unparsed dictionary.
    parameters_in_separate_columns=True,
)
print(df.head(n=10))

####################################################################################################
# We can see all the hyperparameter names in the columns of the dataframe:
for name in df.columns:
    print(name)

####################################################################################################
# Next, we cast and transform the hyperparameters of interest (``C`` and ``gamma``) so that we
# can nicely plot them.
hyperparameters = ["sklearn.svm.classes.SVC(16)_C", "sklearn.svm.classes.SVC(16)_gamma"]
df[hyperparameters] = df[hyperparameters].astype(float).apply(np.log10)

####################################################################################################
# Option 1 - plotting via the pandas helper functions
# ===================================================
#
df.plot.hexbin(
    x="sklearn.svm.classes.SVC(16)_C",
    y="sklearn.svm.classes.SVC(16)_gamma",
    C="value",
    reduce_C_function=np.mean,
    gridsize=25,
    title="SVM performance landscape",
)

####################################################################################################
# Option 2 - plotting via matplotlib
# ==================================
#
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

C = df["sklearn.svm.classes.SVC(16)_C"]
gamma = df["sklearn.svm.classes.SVC(16)_gamma"]
score = df["value"]

# Plotting all evaluations:
ax.plot(C, gamma, "ko", ms=1)
# Create a contour plot
cntr = ax.tricontourf(C, gamma, score, levels=12, cmap="RdBu_r")
# Adjusting the colorbar
fig.colorbar(cntr, ax=ax, label="accuracy")
# Adjusting the axis limits
ax.set(
    xlim=(min(C), max(C)),
    ylim=(min(gamma), max(gamma)),
    xlabel="C (log10)",
    ylabel="gamma (log10)",
)
ax.set_title("SVM performance landscape")
