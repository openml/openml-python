# %% [markdown]
# Tasks define a target and a train/test split, which we can use for benchmarking.

# %%
import openml

# %% [markdown]
# For this tutorial we will use the famous King+Rook versus King+Pawn on A7 dataset, which has
# the dataset ID 3 ([dataset on OpenML](https://www.openml.org/d/3)), and for which there exist
# tasks with all important estimation procedures. It is small enough (less than 5000 samples) to
# efficiently use it in an example.
#
# We will first start with ([task 233](https://www.openml.org/t/233)), which is a task with a
# holdout estimation procedure.

# %%
task_id = 233
task = openml.tasks.get_task(task_id)

# %% [markdown]
# Now that we have a task object we can obtain the number of repetitions, folds and samples as
# defined by the task:

# %%
n_repeats, n_folds, n_samples = task.get_split_dimensions()

# %% [markdown]
# * ``n_repeats``: Number of times the model quality estimation is performed
# * ``n_folds``: Number of folds per repeat
# * ``n_samples``: How many data points to use. This is only relevant for learning curve tasks
#
# A list of all available estimation procedures is available
# [here](https://www.openml.org/search?q=%2520measure_type%3Aestimation_procedure&type=measure).
#
# Task ``233`` is a simple task using the holdout estimation procedure and therefore has only a
# single repeat, a single fold and a single sample size:

# %%
print(
    f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}."
)

# %% [markdown]
# We can now retrieve the train/test split for this combination of repeats, folds and number of
# samples (indexing is zero-based). Usually, one would loop over all repeats, folds and sample
# sizes, but we can neglect this here as there is only a single repetition.

# %%
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0,
    fold=0,
    sample=0,
)

print(train_indices.shape, train_indices.dtype)
print(test_indices.shape, test_indices.dtype)

# %% [markdown]
# And then split the data based on this:

# %%
X, y = task.get_X_and_y()
X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

print(
    f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}"
)

# %% [markdown]
# Obviously, we can also retrieve cross-validation versions of the dataset used in task ``233``:

# %%
task_id = 3
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y()
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}."
)

# %% [markdown]
# And then perform the aforementioned iteration over all splits:

# %%
for repeat_idx in range(n_repeats):
    for fold_idx in range(n_folds):
        for sample_idx in range(n_samples):
            train_indices, test_indices = task.get_train_test_split_indices(
                repeat=repeat_idx,
                fold=fold_idx,
                sample=sample_idx,
            )
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            print(
                f"Repeat #{repeat_idx}, fold #{fold_idx}, samples {sample_idx}: X_train.shape: {X_train.shape}, "
                f"y_train.shape {y_train.shape}, X_test.shape {X_test.shape}, y_test.shape {y_test.shape}"
            )

# %% [markdown]
# And also versions with multiple repeats:

# %%
task_id = 1767
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y()
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}."
)

# %% [markdown]
# And then again perform the aforementioned iteration over all splits:

# %%
for repeat_idx in range(n_repeats):
    for fold_idx in range(n_folds):
        for sample_idx in range(n_samples):
            train_indices, test_indices = task.get_train_test_split_indices(
                repeat=repeat_idx,
                fold=fold_idx,
                sample=sample_idx,
            )
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            print(
                f"Repeat #{repeat_idx}, fold #{fold_idx}, samples {sample_idx}: X_train.shape: {X_train.shape}, "
                f"y_train.shape {y_train.shape}, X_test.shape {X_test.shape}, y_test.shape {y_test.shape}"
            )

# %% [markdown]
# And finally a task based on learning curves:

# %%
task_id = 1702
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y()
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}."
)

# %% [markdown]
# And then again perform the aforementioned iteration over all splits:

# %%
for repeat_idx in range(n_repeats):
    for fold_idx in range(n_folds):
        for sample_idx in range(n_samples):
            train_indices, test_indices = task.get_train_test_split_indices(
                repeat=repeat_idx,
                fold=fold_idx,
                sample=sample_idx,
            )
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            print(
                f"Repeat #{repeat_idx}, fold #{fold_idx}, samples {sample_idx}: X_train.shape: {X_train.shape}, "
                f"y_train.shape {y_train.shape}, X_test.shape {X_test.shape}, y_test.shape {y_test.shape}"
            )
