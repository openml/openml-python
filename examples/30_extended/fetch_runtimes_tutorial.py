"""
=================
Fetching Runtimes
=================

The runtime of machine learning models on specific datasets can be a deciding
factor on the choice of algorithms, especially for benchmarking and comparison
purposes. OpenML provides runtime data from runs of model fit and prediction
on tasks or datasets, for both the CPU-clock as well as the actual wallclock-
time incurred. The objective of this example is to illustrate how to retrieve
such timing measures, and also offer some potential means of usage and
interpretation of the same.

We shall cover these 3 representative scenarios:

* Retrieve runtimes for Random Forest training and prediction on each of the cross-validation folds

* Test the above setting in a parallel setup and monitor the difference using runtimes retrieved

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

task_id = 167119

task = openml.tasks.get_task(task_id)
print(task)

# Viewing associated data
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    "Task {}: number of repeats: {}, number of folds: {}, number of samples {}.".format(
        task_id, n_repeats, n_folds, n_samples,
    )
)

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
clf = RandomForestClassifier(n_estimators=10)

run1 = openml.runs.run_model_on_task(
    model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False,
)
measures = run1.fold_evaluations

# The timing and performance metrics available are
for key in measures.keys():
    print(key)
print()

# The performance metric is recorded under `predictive_accuracy` per fold
# and can be retrieved as:
for repeat, val1 in measures["predictive_accuracy"].items():
    for fold, val2 in val1.items():
        print("Repeat #{}-Fold #{}: {:.4f}".format(repeat, fold, val2))
    print()

################################################################################
# The remaining entries recorded in `measures` are the runtime records
# related as:
# usercpu_time_millis = usercpu_time_millis_training + usercpu_time_millis_testing
# wall_clock_time_millis = wall_clock_time_millis_training + wall_clock_time_millis_testing
#
# The timing measures recorded as `*_millis_training` contain the per
# repeat-fold timing incurred for the executing of the `.fit()` procedure
# of the model. For `usercpu_time_*` the time recorded using `time.process_time()`
# is converted to milliseconds and stored. Similarly, `time.time()` is used
# to record the time entry for `wall_clock_time_*`. The `*_millis_testing` entry
# follows the same procedure but for time taken for the `.predict()` procedure.

# Comparing the CPU and wall-clock training times of the Random Forest model
print_compare_runtimes(measures)

######################################################################
# Case 2: Running a Random Forest model on an OpenML task in parallel
# ********************************************************************
# Redefining the model to allow parallelism with `n_jobs=2` (2 cores)
clf = RandomForestClassifier(n_estimators=10, n_jobs=2)

run2 = openml.runs.run_model_on_task(
    model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False
)
measures = run2.fold_evaluations
# The wall-clock time recorded per fold should be lesser than Case 1 above
print_compare_runtimes(measures)

####################################################################################
# Running a Random Forest model on an OpenML task in parallel (all cores available):

# Redefining the model to use all available cores with `n_jobs=-1`
clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)

run3 = openml.runs.run_model_on_task(
    model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False
)
measures = run3.fold_evaluations
# The wall-clock time recorded per fold should be lesser than the case above,
# if more than 2 CPU cores are available. The speed-up is more pronounced for
# larger datasets.
print_compare_runtimes(measures)

#############################################################################
# The CPU time interpretation becomes ambiguous when jobs are distributed over a
# unknown number of cores. It is difficult to capture both the
# availability of the number of cores/threads and their eventual utilisation from
# OpenML, for various cases that can arise as mentioned below. Therefore, the final
# interpretation of the runtimes are left to the `user`.
#
# It should also be noted that there are multiple levels at which parallelism can
# occur here.
#
# * At the outermost level, OpenML tasks contain fixed data splits, on which the
#   defined model/flow is executed. Thus, a model can be fit on each OpenML dataset fold
#   in parallel using the `n_jobs` parameter to `run_model_on_task` or `run_flow_on_task`.
#
# * The model/flow specified can also include scikit-learn models that perform their own
#   parallelization. For instance, by specifying `n_jobs` in the Random Forest model definition.
#
# * The sklearn model can further be an HPO estimator and contain it's own parallelization.
#   If the base estimator used supports `parallelization`, then there's a 2-level nested definition
#   for parallelization possible. Such an example is explored next.

#####################################################################
# Case 3: Running and benchmarking HPO algorithms with their runtimes
# *******************************************************************
# We shall now optimize a similar RandomForest model for the same task using
# scikit-learn's HPO support by using GridSearchCV to optimize our earlier
# RandomForest model's hyperparameter `n_estimators`. scikit-learn also provides a
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

run4 = openml.runs.run_model_on_task(
    model=grid_pipe, task=task, upload_flow=False, avoid_duplicate_runs=False, n_jobs=2
)
measures = run4.fold_evaluations
print_compare_runtimes(measures)

##################################################################################
# Like any optimisation problem, scikit-learn's HPO estimators also generate
# a sequence of configurations which are evaluated, using which the best found
# configuration is tracked throughout the trace.
# The OpenML run object stores these traces as OpenMLRunTrace objects accessible
# using keys of the pattern (repeat, fold, iterations). Here `fold` implies the
# outer-cross validation fold as obtained from the task data splits in OpenML.
# GridSearchCV here performs grid search over the inner-cross validation folds as
# parameterized by the `cv` parameter. Since `GridSearchCV` in this example performs a
# `2-fold` cross validation, the runtime recorded per repeat-per fold in the run object
# is for the entire `fit()` procedure of GridSearchCV thus subsuming the runtimes of
# the 2-fold (inner) CV search performed.

# We earlier extracted the number of repeats and folds for this task:
print("# repeats: {}\n# folds: {}".format(n_repeats, n_folds))

# To extract the training runtime of the first repeat, first fold:
print(run4.fold_evaluations["wall_clock_time_millis_training"][0][0])

##################################################################################
# To extract the training runtime of the 1-st repeat, 4-th (outer) fold and also
# to fetch the parameters and performance of the evaluations made during
# the 1-st repeat, 4-th fold evaluation by the Grid Search model.

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

##################################################################################
# Scikit-learn's HPO estimators also come with a hyperparameter `refit=True` as a default.
# In our previous model definition it was set to True by default, which meant that the best
# found hyperparameter configuration was used to refit or retrain the model without any inner
# cross validation. This extra refit time measure is provided by the scikit-learn model as the
# attribute `refit_time_`.
# This time is included in the `wall_clock_time_millis_training` measure.
#
# For non-HPO estimators, `wall_clock_time_millis = wall_clock_time_millis_training + wall_clock_time_millis_testing`.
#
# For HPO estimators, `wall_clock_time_millis = wall_clock_time_millis_training + wall_clock_time_millis_testing + refit_time`.
#
# This refit time can therefore be explicitly extracted in this manner:


def extract_refit_time(run, repeat, fold):
    refit_time = (
        run.fold_evaluations["wall_clock_time_millis"][repeat][fold]
        - run.fold_evaluations["wall_clock_time_millis_training"][repeat][fold]
        - run.fold_evaluations["wall_clock_time_millis_testing"][repeat][fold]
    )
    return refit_time


for repeat in range(n_repeats):
    for fold in range(n_folds):
        print(
            "Repeat #{}-Fold #{}: {:.4f}".format(
                repeat, fold, extract_refit_time(run4, repeat, fold)
            )
        )

############################################################################
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
run5 = openml.runs.run_model_on_task(
    model=rs_pipe, task=task, upload_flow=False, avoid_duplicate_runs=False, n_jobs=2
)

################################################################################
# Since for the call to `openml.runs.run_model_on_task` the parameter
# `n_jobs` is set to its default None, the evaluations across the OpenML folds
# are not parallelized. Hence, the time recorded is agnostic to the `n_jobs`
# being set at both the HPO estimator `GridSearchCV` as well as the base
# estimator `RandomForestClassifier` in this case. OpenML only records the
# time taken for the completion of the complete `fit()` call, per-repeat per-fold.
#
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
