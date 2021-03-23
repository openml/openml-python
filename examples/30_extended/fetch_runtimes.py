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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


############################################################################
# Preparing tasks and scikit-learn models
# ***************************************

task_id = 1792  # 1778  # supervised classification on mfeat-fourier
task = openml.tasks.get_task(task_id)
print(task)

# Viewing associated data
X, y = task.get_X_and_y(dataset_format="dataframe")
print(X.head)
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
                "Repeat #{}-Fold#{}: CPU-{:.3f} vs Wall-{:.3f}".format(
                    repeat, fold, val2, measures["wall_clock_time_millis_training"][repeat][fold]
                )
            )
        print()


############################################################################
# Case 1: Running a Random Forest model on an OpenML task
# *******************************************************
# We'll run a Random Forest model and obtain an OpenML run-object. We can
# see the evaluations recorded per fold for the dataset and the information
# available for this run.

run1 = openml.runs.run_model_on_task(model=clf, task=task)
measures = run1.fold_evaluations

# The timing and performance metrics available are
for key in measures.keys():
    print(key)

# The performance metric is recorded under `predictive_accuracy` per fold
# and can be retrieved as:
for repeat, val1 in measures["predictive_accuracy"].items():
    for fold, val2 in val1.items():
        print("Repeat #{}-Fold#{}: {}".format(repeat, fold, val2))
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
# Running the same model on the same task but with parallelism (n_jobs > 1)

run2 = openml.runs.run_model_on_task(model=clf, task=task, n_jobs=4)
measures = run2.fold_evaluations

# Comparing the CPU and wall-clock training times of the Random Forest model
print_compare_runtimes(measures)

############################################################################
# Case 3: Running and benchmarking HPO algorithms with their runtimes
# *******************************************************************
# We shall now optimize the same RandomForest model for the same task using
# scikit-learn HPO support by using GridSearchCV to optimize our earlier
# RandomForest model's parameter `n_estimators`. sklearn also provides a
# `refit_time_` for such HPO models, i.e., the time incurred by training
# and evaluating the model on the best found parameter setting. OpenML
# includes this in the `wall_clock_time_millis_training` measure recorded.

# GridSearchCV model
grid_pipe = GridSearchCV(
    estimator=clf,
    param_grid={"n_estimators": np.linspace(start=2, stop=100, num=10).astype(int).tolist()},
    cv=10,
)

run3 = openml.runs.run_model_on_task(model=grid_pipe, task=task, n_jobs=4)
measures = run3.fold_evaluations
print_compare_runtimes(measures)
