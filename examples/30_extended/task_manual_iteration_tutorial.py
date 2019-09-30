"""
Tasks: retrieving splits
========================

Tasks define a target and a train/test split. Normally, they are the input to the function
``openml.runs.run_model_on_task`` which automatically runs the model on all splits of the task.
However, sometimes it is necessary to manually split a dataset to perform experiments outside of
the functions provided by OpenML. One such example is in the benchmark library
`HPOlib2 <https://github.com/automl/hpolib2>`_ which extensively uses data from OpenML,
but not OpenML's functionality to conduct runs.
"""

import openml

####################################################################################################
task_id = 233
task = openml.tasks.get_task(task_id)

####################################################################################################
# Now that we have a task object we can obtain the number of repetitions, folds and samples as
# defined by the task:

n_repeats, n_folds, n_samples = task.get_split_dimensions()

####################################################################################################
# * ``n_repeats``: Number of times the model quality estimation is performed
# * ``n_folds``: Number of folds per repeat
# * ``n_samples``: How many data points to use. This is only relevant for learning curve tasks
#
# A list of all available estimation procedures is available
# `here <https://www.openml.org/search?q=%2520measure_type%3Aestimation_procedure&type=measure>`_.
#
# Task ``233`` is a simple task using the holdout estimation procedure and therefore has only a
# single repeat, a single fold and a single sample size:

print(n_repeats, n_folds, n_samples)

####################################################################################################
# We can now retrieve the train/test split for this combination of repeats, folds and number of
# samples (indexing is zero-based):

train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0,
    fold=0,
    sample=0,
)

print(train_indices.shape, train_indices.dtype)
print(test_indices.shape, test_indices.dtype)

####################################################################################################
# And then split the data based on this:

X, y, _, _ = task.get_dataset().get_data(task.target_name)
X_train = X.loc[train_indices]
y_train = y[train_indices]
X_test = X.loc[test_indices]
y_test = y[test_indices]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

####################################################################################################
# Obviously, we can also retrieve cross-validation versions of the dataset used in task ``233``:

task_id = 3
task = openml.tasks.get_task(task_id)
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(n_repeats, n_folds, n_samples)

####################################################################################################
# And also versions with multiple repeats:

task_id = 1767
task = openml.tasks.get_task(task_id)
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(n_repeats, n_folds, n_samples)

####################################################################################################
# And finally a task based on learning curves:

task_id = 1702
task = openml.tasks.get_task(task_id)
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(n_repeats, n_folds, n_samples)
