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


openml.config.apikey = "7ff6d43e00cc810a00c01bd770996dfc"

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
n_iter = 5
grid_pipe = GridSearchCV(
    estimator=clf,
    param_grid={"n_estimators": np.linspace(start=1, stop=50, num=n_iter).astype(int).tolist()},
    cv=2,
    n_jobs=2,
)

run4 = openml.runs.run_model_on_task(model=grid_pipe, task=task, n_jobs=2)
measures = run4.fold_evaluations
print_compare_runtimes(measures)
print()

# Like any optimisation problem, scikit-learn's HPO estimators also generate
# a sequence of configurations which are evaluated as the best found
# configuration is tracked throughout the trace.
# The OpenML run object stores these traces as OpenMLRunTrace objects accessible
# using keys of the pattern (repeat, fold, iterations). Since `iterations` is part
# of the scikit-learn model here, the runtime recorded per repeat-per fold is for
# this entire `fit()` procedure.

# We earlier extracted the number of repeats and folds for this task:
print("# repeats: {}\n# fold: {}".format(n_repeats, n_folds))

# To extract the training runtime of the first repeat, first fold:
print(run4.fold_evaluations["wall_clock_time_millis_training"][0][0])

# To extract the training runtime of the 1-st repeat, 4-th fold and also
# to fetch the parameters and performance of the evaluations made during
# the 1-st repeat, 4-th fold evaluation by the Grid Search model
_repeat = 0
_fold = 3
print(
    "Total runtime for repeat {}'s fold {}: {:4f} ms".format(
        _repeat, _fold, run4.fold_evaluations["wall_clock_time_millis_training"][_repeat][_fold]
    )
)
for i in range(n_iter):
    key = (_repeat, _fold, i)
    r = run4.trace.trace_iterations[key]
    print(
        "n_estimators: {:>2} - score: {:.3f}".format(
            r.parameters["parameter_n_estimators"], r.evaluation
        )
    )

# Along with the GridSearchCV already used above, we demonstrate how such
# optimisation traces can be retrieved by showing an application of these
# traces - comparing the speed of finding the best configuration using
# RandomizedSearchCV and GridSearchCV available with scikit-learn.

# RandomizedSearchCV model
rs_pipe = RandomizedSearchCV(
    estimator=clf,
    param_distributions={
        "n_estimators": np.linspace(start=1, stop=50, num=15).astype(int).tolist()
    },
    cv=2,
    n_iter=n_iter,
    n_jobs=2,
)
run5 = openml.runs.run_model_on_task(model=rs_pipe, task=task)


# Since for the call to `openml.runs.run_model_on_task` the parameter
# `n_jobs` is set to its default None, the evaluations across the OpenML folds
# are not parallelized. Hence, the time recorded is agnostic to the `n_jobs`
# being set at both the HPO estimator `GridSearchCV` as well as the base
# estimator `RandomForestClassifier` in this case. OpenML only records the
# time taken for the completion of the complete `fit()` call, per-repeat per-fold.

# This notion can be used to extract and plot the best found performance per
# fold by the HPO model and the corresponding time taken for search across
# that fold. Moreover, since `n_jobs=None` for `openml.runs.run_model_on_task`
# the runtimes per fold can be cumulatively added to plot the trace against time.


def extract_trace_data(run, n_repeats, n_folds, n_iter, key=None):
    key = "wall_clock_time_millis_training" if key is None else key
    data = {"score": [], "runtime": []}
    for i_r in range(n_repeats):
        for i_f in range(n_folds):
            data["runtime"].append(run.fold_evaluations[key][i_r][i_f])
            for i_i in range(n_iter):
                r = run.trace.trace_iterations[(i_r, i_f, i_i)]
                if r.selected:
                    data["score"].append(r.evaluation)
                    break
    return data


def get_incumbent_trace(trace):
    best_score = 1
    inc_trace = []
    for i, r in enumerate(trace):
        if i == 0 or (1 - r) < best_score:
            best_score = 1 - r
        inc_trace.append(best_score)
    return inc_trace


grid_data = extract_trace_data(run4, n_repeats, n_folds, n_iter)
rs_data = extract_trace_data(run5, n_repeats, n_folds, n_iter)

plt.clf()
plt.plot(
    np.cumsum(grid_data["runtime"]), get_incumbent_trace(grid_data["score"]), label="Grid Search"
)
plt.plot(
    np.cumsum(rs_data["runtime"]), get_incumbent_trace(rs_data["score"]), label="Random Search"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Wallclock time (in milliseconds)")
plt.ylabel("1 - Accuracy")
plt.title("Optimisation Trace Comparison")
plt.legend()
plt.show()
