"""
================================
Plotting hyperparameter surfaces
================================
"""
import openml
import numpy as np

# Choose an SVM flow, for example 8353, and a task.
df = openml.evaluations.list_evaluations_setups(
    function='predictive_accuracy',
    flow=[8353],
    task=[6],
    output_format='dataframe',
    parameters_in_separate_columns=True,
)
hyperparameters = ['sklearn.svm.classes.SVC(16)_C', 'sklearn.svm.classes.SVC(16)_gamma']
df[hyperparameters] = df[hyperparameters].astype(float).apply(np.log)

####################################################################################################
# Option 1 - plotting via the pandas helper functions
# ===================================================
#
df.plot.hexbin(
    x='sklearn.svm.classes.SVC(16)_C',
    y='sklearn.svm.classes.SVC(16)_gamma',
    C='value', reduce_C_function=np.mean, gridsize=25,
)

####################################################################################################
# Option 2 - plotting via matplotlib
# ==================================
#
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

C = df['sklearn.svm.classes.SVC(16)_C']
gamma = df['sklearn.svm.classes.SVC(16)_gamma']
score = df['value']

# Plotting all evaluations:
ax.plot(C, gamma, 'ko', ms=1)
# Create a contour plot
cntr = ax.tricontourf(C, gamma, score, levels=12, cmap="RdBu_r")
# Adjusting the colorbar
fig.colorbar(cntr, ax=ax, label="accuracy")
# Adjusting the axis limits
ax.set(
    xlim=[min(C),max(C)],
    ylim=[min(gamma),max(gamma)],
    xlabel="C (log10)",
    ylabel="gamma (log10)",
)

####################################################################################################
# Option 3 - exact code example from the OpenML-Python paper
# ==========================================================
#

import openml
import numpy as np
import matplotlib.pyplot as plt
df = openml.evaluations.list_evaluations_setups(
    'predictive_accuracy', flow=[8353], task=[6],
    output_format='dataframe', parameters_in_separate_columns=True,
) # Choose an SVM flow, for example 8353, and a task.
hp_names = ['sklearn.svm.classes.SVC(16)_C','sklearn.svm.classes.SVC(16)_gamma']
df[hp_names] = df[hp_names].astype(float).apply(np.log)
C, gamma, score = df[hp_names[0]], df[hp_names[1]], df['value']
cntr = plt.tricontourf(C, gamma, score, levels=12, cmap="RdBu_r")
plt.colorbar(cntr, label="accuracy")
plt.xlim((min(C), max(C))); plt.ylim((min(gamma), max(gamma)))
plt.xlabel("C (log10)"); plt.ylabel("gamma (log10)")
