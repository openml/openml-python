"""

==========================================
Measuring runtimes for Scikit-learn models
==========================================

The runtime of machine learning models on specific datasets can be a deciding
factor on the choice of algorithms, especially for benchmarking and comparison
purposes. OpenML's scikit-learn extension provides runtime data from runs of
model fit and prediction on tasks or datasets, for both the CPU-clock as well
as the actual wallclock-time incurred. The objective of this example is to
illustrate how to retrieve such timing measures, and also offer some potential
means of usage and interpretation of the same.

It should be noted that there are multiple levels at which parallelism can occur.

* At the outermost level, OpenML tasks contain fixed data splits, on which the
  defined model/flow is executed. Thus, a model can be fit on each OpenML dataset fold
  in parallel using the `n_jobs` parameter to `run_model_on_task` or `run_flow_on_task`
  (illustrated under Case 2 & 3 below).

* The model/flow specified can also include scikit-learn models that perform their own
  parallelization. For instance, by specifying `n_jobs` in a Random Forest model definition
  (covered under Case 2 below).

* The sklearn model can further be an HPO estimator and contain it's own parallelization.
  If the base estimator used also supports `parallelization`, then there's at least a 2-level nested
  definition for parallelization possible (covered under Case 3 below).

We shall cover these 5 representative scenarios for:

* (Case 1) Retrieving runtimes for Random Forest training and prediction on each of the
  cross-validation folds

* (Case 2) Testing the above setting in a parallel setup and monitor the difference using
  runtimes retrieved

* (Case 3) Comparing RandomSearchCV and GridSearchCV on the above task based on runtimes

* (Case 4) Running models that don't run in parallel or models which scikit-learn doesn't
  parallelize

* (Case 5) Running models that do not release the Python Global Interpreter Lock (GIL)
"""

############################################################################

# License: BSD 3-Clause

import openml
import numpy as np
from matplotlib import pyplot as plt
from joblib.parallel import parallel_backend

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
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
        task_id,
        n_repeats,
        n_folds,
        n_samples,
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


############################################################################
# Case 1: Running a Random Forest model on an OpenML task
# *******************************************************
# We'll run a Random Forest model and obtain an OpenML run object. We can
# see the evaluations recorded per fold for the dataset and the information
# available for this run.

clf = RandomForestClassifier(n_estimators=10)

run1 = openml.runs.run_model_on_task(
    model=clf,
    task=task,
    upload_flow=False,
    avoid_duplicate_runs=False,
)
measures = run1.fold_evaluations

print("The timing and performance metrics available: ")
for key in measures.keys():
    print(key)
print()

print(
    "The performance metric is recorded under `predictive_accuracy` per "
    "fold and can be retrieved as: "
)
for repeat, val1 in measures["predictive_accuracy"].items():
    for fold, val2 in val1.items():
        print("Repeat #{}-Fold #{}: {:.4f}".format(repeat, fold, val2))
    print()

################################################################################
# The remaining entries recorded in `measures` are the runtime records
# related as:
#
# usercpu_time_millis = usercpu_time_millis_training + usercpu_time_millis_testing
#
# wall_clock_time_millis = wall_clock_time_millis_training + wall_clock_time_millis_testing
#
# The timing measures recorded as `*_millis_training` contain the per
# repeat-per fold timing incurred for the execution of the `.fit()` procedure
# of the model. For `usercpu_time_*` the time recorded using `time.process_time()`
# is converted to `milliseconds` and stored. Similarly, `time.time()` is used
# to record the time entry for `wall_clock_time_*`. The `*_millis_testing` entry
# follows the same procedure but for time taken for the `.predict()` procedure.

# Comparing the CPU and wall-clock training times of the Random Forest model
print_compare_runtimes(measures)

######################################################################
# Case 2: Running Scikit-learn model on an OpenML task in parallel
# ****************************************************************
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

####################################################################################
# We can now observe that the ratio of CPU time to wallclock time is lower
# than in case 1. This happens because joblib by default spawns subprocesses
# for the workloads for which CPU time cannot be tracked. Therefore, interpreting
# the reported CPU and wallclock time requires knowledge of the parallelization
# applied at runtime.

####################################################################################
# Running the same task with a different parallel backend. Joblib provides multiple
# backends: {`loky` (default), `multiprocessing`, `dask`, `threading`, `sequential`}.
# The backend can be explicitly set using a joblib context manager. The behaviour of
# the job distribution can change and therefore the scale of runtimes recorded too.

with parallel_backend(backend="multiprocessing", n_jobs=-1):
    run3_ = openml.runs.run_model_on_task(
        model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False
    )
measures = run3_.fold_evaluations
print_compare_runtimes(measures)

####################################################################################
# The CPU time interpretation becomes ambiguous when jobs are distributed over an
# unknown number of cores or when subprocesses are spawned for which the CPU time
# cannot be tracked, as in the examples above. It is impossible for OpenML-Python
# to capture the availability of the number of cores/threads, their eventual
# utilisation and whether workloads are executed in subprocesses, for various
# cases that can arise as demonstrated in the rest of the example. Therefore,
# the final interpretation of the runtimes is left to the `user`.

#####################################################################
# Case 3: Running and benchmarking HPO algorithms with their runtimes
# *******************************************************************
# We shall now optimize a similar RandomForest model for the same task using
# scikit-learn's HPO support by using GridSearchCV to optimize our earlier
# RandomForest model's hyperparameter `n_estimators`. Scikit-learn also provides a
# `refit_time_` for such HPO models, i.e., the time incurred by training
# and evaluating the model on the best found parameter setting. This is
# included in the `wall_clock_time_millis_training` measure recorded.

from sklearn.model_selection import GridSearchCV


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
# Scikit-learn's HPO estimators also come with an argument `refit=True` as a default.
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
# Since for the call to ``openml.runs.run_model_on_task`` the parameter
# ``n_jobs`` is set to its default ``None``, the evaluations across the OpenML folds
# are not parallelized. Hence, the time recorded is agnostic to the ``n_jobs``
# being set at both the HPO estimator ``GridSearchCV`` as well as the base
# estimator ``RandomForestClassifier`` in this case. The OpenML extension only records the
# time taken for the completion of the complete ``fit()`` call, per-repeat per-fold.
#
# This notion can be used to extract and plot the best found performance per
# fold by the HPO model and the corresponding time taken for search across
# that fold. Moreover, since ``n_jobs=None`` for ``openml.runs.run_model_on_task``
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

################################################################################
# Case 4: Running models that scikit-learn doesn't parallelize
# *************************************************************
# Both scikit-learn and OpenML depend on parallelism implemented through `joblib`.
# However, there can be cases where either models cannot be parallelized or don't
# depend on joblib for its parallelism. 2 such cases are illustrated below.
#
# Running a Decision Tree model that doesn't support parallelism implicitly, but
# using OpenML to parallelize evaluations for the outer-cross validation folds.

dt = DecisionTreeClassifier()

run6 = openml.runs.run_model_on_task(
    model=dt, task=task, upload_flow=False, avoid_duplicate_runs=False, n_jobs=2
)
measures = run6.fold_evaluations
print_compare_runtimes(measures)

################################################################################
# Although the decision tree does not run in parallel, it can release the
# `Python GIL <https://docs.python.org/dev/glossary.html#term-global-interpreter-lock>`_.
# This can result in surprising runtime measures as demonstrated below:

with parallel_backend("threading", n_jobs=-1):
    run7 = openml.runs.run_model_on_task(
        model=dt, task=task, upload_flow=False, avoid_duplicate_runs=False
    )
measures = run7.fold_evaluations
print_compare_runtimes(measures)

################################################################################
# Running a Neural Network from scikit-learn that uses scikit-learn independent
# parallelism using libraries such as `MKL, OpenBLAS or BLIS
# <https://scikit-learn.org/stable/computing/parallelism.html#parallel-numpy-and-scipy-routines-from-numerical-libraries>`_.

mlp = MLPClassifier(max_iter=10)

run8 = openml.runs.run_model_on_task(
    model=mlp, task=task, upload_flow=False, avoid_duplicate_runs=False
)
measures = run8.fold_evaluations
print_compare_runtimes(measures)

################################################################################
# Case 5: Running Scikit-learn models that don't release GIL
# **********************************************************
# Certain Scikit-learn models do not release the `Python GIL
# <https://docs.python.org/dev/glossary.html#term-global-interpreter-lock>`_ and
# are also not executed in parallel via a BLAS library. In such cases, the
# CPU times and wallclock times are most likely trustworthy. Note however
# that only very few models such as naive Bayes models are of this kind.

clf = GaussianNB()

with parallel_backend("multiprocessing", n_jobs=-1):
    run9 = openml.runs.run_model_on_task(
        model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False
    )
measures = run9.fold_evaluations
print_compare_runtimes(measures)

################################################################################
# Summmary
# *********
# The scikit-learn extension for OpenML-Python records model runtimes for the
# CPU-clock and the wall-clock times. The above examples illustrated how these
# recorded runtimes can be extracted when using a scikit-learn model and under
# parallel setups too. To summarize, the scikit-learn extension measures the:
#
# * `CPU-time` & `wallclock-time` for the whole run
#
#   * A run here corresponds to a call to `run_model_on_task` or `run_flow_on_task`
#   * The recorded time is for the model fit for each of the outer-cross validations folds,
#     i.e., the OpenML data splits
#
# * Python's `time` module is used to compute the runtimes
#
#   * `CPU-time` is recorded using the responses of `time.process_time()`
#   * `wallclock-time` is recorded using the responses of `time.time()`
#
# * The timings recorded by OpenML per outer-cross validation fold is agnostic to
#   model parallelisation
#
#   * The wallclock times reported in Case 2 above highlights the speed-up on using `n_jobs=-1`
#     in comparison to `n_jobs=2`, since the timing recorded by OpenML is for the entire
#     `fit()` procedure, whereas the parallelisation is performed inside `fit()` by scikit-learn
#   * The CPU-time for models that are run in parallel can be difficult to interpret
#
# * `CPU-time` & `wallclock-time` for each search per outer fold in an HPO run
#
#   * Reports the total time for performing search on each of the OpenML data split, subsuming
#     any sort of parallelism that happened as part of the HPO estimator or the underlying
#     base estimator
#   * Also allows extraction of the `refit_time` that scikit-learn measures using `time.time()`
#     for retraining the model per outer fold, for the best found configuration
#
# * `CPU-time` & `wallclock-time` for models that scikit-learn doesn't parallelize
#
#   * Models like Decision Trees or naive Bayes don't parallelize and thus both the wallclock and
#     CPU times are similar in runtime for the OpenML call
#   * However, models implemented in Cython, such as the Decision Trees can release the GIL and
#     still run in parallel if a `threading` backend is used by joblib.
#   * Scikit-learn Neural Networks can undergo parallelization implicitly owing to thread-level
#     parallelism involved in the linear algebraic operations and thus the wallclock-time and
#     CPU-time can differ.
#
# Because of all the cases mentioned above it is crucial to understand which case is triggered
# when reporting runtimes for scikit-learn models measured with OpenML-Python!
