"""
=================
Fetching Runtimes
=================

The runtime of machine learning models on specific datasets can be a deciding
factor on the choice of algorithms. Especially for benchmarking and comparison
purposes. OpenML provides runtime data from runs of model fit and prediction
on tasks or datasets, for both the CPU-clock as well as the actual wallclock-
time incurred. The objective of this example is to illustrate how to retrieve
such timing measures, and also offer some potential means of usage and
interpretation of the same.

We shall cover these 3 representative scenarios:

* Retrieve runtimes for Random Forest training and prediction on each of the
cross-validation folds
* Test the above setting in a parallel setup and monitor the difference using
runtimes retrieved
* Compare RandomSearchCV and GridSearchCV on the above task based on runtimes
"""

############################################################################

# License: BSD 3-Clause

import openml
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


############################################################################
# Preparing tasks and scikit-learn models
# ***************************************

task_id = 168908

task = openml.tasks.get_task(task_id)
print(task)

# Viewing associated data
X, y = task.get_X_and_y(dataset_format="array")
# print(X.head)
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    "Task {}: number of repeats: {}, number of folds: {}, number of samples {}.".format(
        task_id, n_repeats, n_folds, n_samples,
    )
)

clf = RandomForestClassifier(n_estimators=10)

# Creating utility function
def print_compare_runtimes(measures):
    for repeat, val1 in measures["usercpu_time_millis_training"].items():
        for fold, val2 in val1.items():
            print(
                "Repeat #{}-Fold #{}: CPU-{:.3f} vs Wall-{:.3f}".format(
                    repeat, fold, val2, measures["wall_clock_time_millis_training"][repeat][fold]
                )
            )
        print()


############################################################################
# Case 1: Running a Random Forest model on an OpenML task
# *******************************************************
# We'll run a Random Forest model and obtain an OpenML run object. We can
# see the evaluations recorded per fold for the dataset and the information
# available for this run.

run1 = openml.runs.run_model_on_task(model=clf, task=task, dataset_format="array")
measures = run1.fold_evaluations

# The timing and performance metrics available are
for key in measures.keys():
    print(key)

# The performance metric is recorded under `predictive_accuracy` per fold
# and can be retrieved as:
for repeat, val1 in measures["predictive_accuracy"].items():
    for fold, val2 in val1.items():
        print("Repeat #{}-Fold #{}: {:.4f}".format(repeat, fold, val2))
    print()

# The remaining entries recorded in `measures` are the runtime records
# related as:
# usercpu_time_millis = usercpu_time_millis_training + usercpu_time_millis_testing
# wall_clock_time_millis = wall_clock_time_millis_training + wall_clock_time_millis_testing
#
# The timing measures recorded as `*_millis_training` contain the per
# repeat-fold timing incurred for the executing of the `.fit()` procedure
# of the model. For `usercpu_time_*` the time recorded using `time.process_time()`
# is converted to milliseconds and stored. Similarly, `time.time()` is used
# to record the time entry for `wall_clock_time_*`. *_millis_testing`
# follows the same procedure but for time taken for the `.predict()` procedure.

# Comparing the CPU and wall-clock training times of the Random Forest model
print_compare_runtimes(measures)

############################################################################
# Case 2: Running a Random Forest model on an OpenML task in parallel
# *******************************************************************
# Redefining the model to allow parallelism with `n_jobs=2`
clf = RandomForestClassifier(n_estimators=10, n_jobs=2)

run2 = openml.runs.run_model_on_task(model=clf, task=task, dataset_format="array")
measures = run2.fold_evaluations
# The wall-clock time recorded per fold should be lesser than Case 1 above
print_compare_runtimes(measures)

# Redefining the model to use all available cores with `n_jobs=-1`
clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)

run3 = openml.runs.run_model_on_task(model=clf, task=task, dataset_format="array")
measures = run3.fold_evaluations
# The wall-clock time recorded per fold should be lesser than the case above,
# if more than 2 CPU cores are available
print_compare_runtimes(measures)

# It should be noted that there are multiple levels at which parallelism can
# occur here.

############################################################################
# Case 3: Running and benchmarking HPO algorithms with their runtimes
# *******************************************************************
# We shall now optimize the same RandomForest model for the same task using
# scikit-learn HPO support by using GridSearchCV to optimize our earlier
# RandomForest model's parameter `n_estimators`. sklearn also provides a
# `refit_time_` for such HPO models, i.e., the time incurred by training
# and evaluating the model on the best found parameter setting. OpenML
# includes this in the `wall_clock_time_millis_training` measure recorded.

clf = RandomForestClassifier(n_estimators=10, n_jobs=2)
# GridSearchCV model
grid_pipe = GridSearchCV(
    estimator=clf,
    param_grid={"n_estimators": np.linspace(start=1, stop=50, num=6).astype(int).tolist()},
    cv=2,
)

run4 = openml.runs.run_model_on_task(model=grid_pipe, task=task, n_jobs=2)
measures = run4.fold_evaluations
print_compare_runtimes(measures)
print()

# Scikit-learn HPO estimators often have a `refit=True` parameter set by default,
# and thus records the time to refit the model with the best found parameters.
# This can be extracted in the following manner:
for repeat, val1 in measures["refit_time"].items():
    for fold, val2 in val1.items():
        print("Repeat #{}-Fold #{}: {:.4f}".format(repeat, fold, val2))
    print()

# Like any optimisation problem, scikit-learn's HPO estimators also generate
# a sequence of configurations which are evaluated as the best found
# configuration is tracked throughout the trace. Along with the GridSearchCV
# already used above, we demonstrate how such optimisation traces can be
# retrieved by showing an application of these traces - comparing the
# speed of finding the best configuration using RandomizedSearchCV and
# GridSearchCV available with scikit-learn.

# RandomizedSearchCV model
rs_pipe = RandomizedSearchCV(
    estimator=clf,
    param_distributions={
        "n_estimators": np.linspace(start=1, stop=50, num=15).astype(int).tolist()
    },
    cv=2,
    n_iter=6,
)
run5 = openml.runs.run_model_on_task(model=rs_pipe, task=task, n_jobs=2)

# This function tracks the best so-far seen accuracy as the regret from 100% accuracy
def get_incumbent_trace(trace):
    # `trace` is an OpenMLRunTrace object
    best_score = 1
    inc_trace = []
    for i, r in enumerate(trace.trace_iterations.values()):
        # `r` is an OpenMLTraceIteration object
        if i == 0 or (r.selected and (1 - r.evaluation) < best_score):
            best_score = 1 - r.evaluation
        inc_trace.append(best_score)
    return inc_trace


plt.clf()
plt.plot(get_incumbent_trace(run4.trace), label="Grid Search")
plt.plot(get_incumbent_trace(run5.trace), label="Random Search")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of evaluations")
plt.ylabel("1 - Accuracy")
plt.title("Optimisation Trace Comparison")
plt.legend()
plt.show()
