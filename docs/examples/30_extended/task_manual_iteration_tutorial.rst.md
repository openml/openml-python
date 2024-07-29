::::: only
html

:::: {.note .sphx-glr-download-link-note}
::: title
Note
:::

`Go to the end <sphx_glr_download_examples_30_extended_task_manual_iteration_tutorial.py>`{.interpreted-text
role="ref"} to download the full example code
::::
:::::

::: rst-class
sphx-glr-example-title
:::

# Tasks: retrieving splits {#sphx_glr_examples_30_extended_task_manual_iteration_tutorial.py}

Tasks define a target and a train/test split. Normally, they are the
input to the function `openml.runs.run_model_on_task` which
automatically runs the model on all splits of the task. However,
sometimes it is necessary to manually split a dataset to perform
experiments outside of the functions provided by OpenML. One such
example is in the benchmark library
[HPOBench](https://github.com/automl/HPOBench) which extensively uses
data from OpenML, but not OpenML\'s functionality to conduct runs.

``` default
# License: BSD 3-Clause

import openml
```

For this tutorial we will use the famous King+Rook versus King+Pawn on
A7 dataset, which has the dataset ID 3 ([dataset on
OpenML](https://www.openml.org/d/3)), and for which there exist tasks
with all important estimation procedures. It is small enough (less than
5000 samples) to efficiently use it in an example.

We will first start with ([task 233](https://www.openml.org/t/233)),
which is a task with a holdout estimation procedure.

``` default
task_id = 233
task = openml.tasks.get_task(task_id)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
```
:::

Now that we have a task object we can obtain the number of repetitions,
folds and samples as defined by the task:

``` default
n_repeats, n_folds, n_samples = task.get_split_dimensions()
```

-   `n_repeats`: Number of times the model quality estimation is
    performed
-   `n_folds`: Number of folds per repeat
-   `n_samples`: How many data points to use. This is only relevant for
    learning curve tasks

A list of all available estimation procedures is available
[here](https://www.openml.org/search?q=%2520measure_type%3Aestimation_procedure&type=measure).

Task `233` is a simple task using the holdout estimation procedure and
therefore has only a single repeat, a single fold and a single sample
size:

``` default
print(
    "Task {}: number of repeats: {}, number of folds: {}, number of samples {}.".format(
        task_id,
        n_repeats,
        n_folds,
        n_samples,
    )
)
```

::: rst-class
sphx-glr-script-out

``` none
Task 233: number of repeats: 1, number of folds: 1, number of samples 1.
```
:::

We can now retrieve the train/test split for this combination of
repeats, folds and number of samples (indexing is zero-based). Usually,
one would loop over all repeats, folds and sample sizes, but we can
neglect this here as there is only a single repetition.

``` default
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0,
    fold=0,
    sample=0,
)

print(train_indices.shape, train_indices.dtype)
print(test_indices.shape, test_indices.dtype)
```

::: rst-class
sphx-glr-script-out

``` none
(2142,) int32
(1054,) int32
```
:::

And then split the data based on this:

``` default
X, y = task.get_X_and_y(dataset_format="dataframe")
X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

print(
    "X_train.shape: {}, y_train.shape: {}, X_test.shape: {}, y_test.shape: {}".format(
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )
)
```

::: rst-class
sphx-glr-script-out

``` none
X_train.shape: (2142, 36), y_train.shape: (2142,), X_test.shape: (1054, 36), y_test.shape: (1054,)
```
:::

Obviously, we can also retrieve cross-validation versions of the dataset
used in task `233`:

``` default
task_id = 3
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y(dataset_format="dataframe")
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    "Task {}: number of repeats: {}, number of folds: {}, number of samples {}.".format(
        task_id,
        n_repeats,
        n_folds,
        n_samples,
    )
)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Task 3: number of repeats: 1, number of folds: 10, number of samples 1.
```
:::

And then perform the aforementioned iteration over all splits:

``` default
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
                "Repeat #{}, fold #{}, samples {}: X_train.shape: {}, "
                "y_train.shape {}, X_test.shape {}, y_test.shape {}".format(
                    repeat_idx,
                    fold_idx,
                    sample_idx,
                    X_train.shape,
                    y_train.shape,
                    X_test.shape,
                    y_test.shape,
                )
            )
```

::: rst-class
sphx-glr-script-out

``` none
Repeat #0, fold #0, samples 0: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 0: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 0: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 0: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 0: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 0: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #6, samples 0: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 0: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 0: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 0: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
```
:::

And also versions with multiple repeats:

``` default
task_id = 1767
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y(dataset_format="dataframe")
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    "Task {}: number of repeats: {}, number of folds: {}, number of samples {}.".format(
        task_id,
        n_repeats,
        n_folds,
        n_samples,
    )
)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Task 1767: number of repeats: 5, number of folds: 2, number of samples 1.
```
:::

And then again perform the aforementioned iteration over all splits:

``` default
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
                "Repeat #{}, fold #{}, samples {}: X_train.shape: {}, "
                "y_train.shape {}, X_test.shape {}, y_test.shape {}".format(
                    repeat_idx,
                    fold_idx,
                    sample_idx,
                    X_train.shape,
                    y_train.shape,
                    X_test.shape,
                    y_test.shape,
                )
            )
```

::: rst-class
sphx-glr-script-out

``` none
Repeat #0, fold #0, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #0, fold #1, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #1, fold #0, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #1, fold #1, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #2, fold #0, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #2, fold #1, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #3, fold #0, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #3, fold #1, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #4, fold #0, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
Repeat #4, fold #1, samples 0: X_train.shape: (1598, 36), y_train.shape (1598,), X_test.shape (1598, 36), y_test.shape (1598,)
```
:::

And finally a task based on learning curves:

``` default
task_id = 1702
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y(dataset_format="dataframe")
n_repeats, n_folds, n_samples = task.get_split_dimensions()
print(
    "Task {}: number of repeats: {}, number of folds: {}, number of samples {}.".format(
        task_id,
        n_repeats,
        n_folds,
        n_samples,
    )
)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Task 1702: number of repeats: 1, number of folds: 10, number of samples 12.
```
:::

And then again perform the aforementioned iteration over all splits:

``` default
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
                "Repeat #{}, fold #{}, samples {}: X_train.shape: {}, "
                "y_train.shape {}, X_test.shape {}, y_test.shape {}".format(
                    repeat_idx,
                    fold_idx,
                    sample_idx,
                    X_train.shape,
                    y_train.shape,
                    X_test.shape,
                    y_test.shape,
                )
            )
```

::: rst-class
sphx-glr-script-out

``` none
Repeat #0, fold #0, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #0, samples 11: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #1, samples 11: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #2, samples 11: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #3, samples 11: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #4, samples 11: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #5, samples 11: X_train.shape: (2876, 36), y_train.shape (2876,), X_test.shape (320, 36), y_test.shape (320,)
Repeat #0, fold #6, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #6, samples 11: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #7, samples 11: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #8, samples 11: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 0: X_train.shape: (64, 36), y_train.shape (64,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 1: X_train.shape: (91, 36), y_train.shape (91,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 2: X_train.shape: (128, 36), y_train.shape (128,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 3: X_train.shape: (181, 36), y_train.shape (181,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 4: X_train.shape: (256, 36), y_train.shape (256,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 5: X_train.shape: (362, 36), y_train.shape (362,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 6: X_train.shape: (512, 36), y_train.shape (512,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 7: X_train.shape: (724, 36), y_train.shape (724,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 8: X_train.shape: (1024, 36), y_train.shape (1024,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 9: X_train.shape: (1448, 36), y_train.shape (1448,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 10: X_train.shape: (2048, 36), y_train.shape (2048,), X_test.shape (319, 36), y_test.shape (319,)
Repeat #0, fold #9, samples 11: X_train.shape: (2877, 36), y_train.shape (2877,), X_test.shape (319, 36), y_test.shape (319,)
```
:::

::: rst-class
sphx-glr-timing

**Total running time of the script:** ( 0 minutes 3.602 seconds)
:::

::::::: {#sphx_glr_download_examples_30_extended_task_manual_iteration_tutorial.py}
:::::: only
html

::::: {.container .sphx-glr-footer .sphx-glr-footer-example}
::: {.container .sphx-glr-download .sphx-glr-download-python}
`Download Python source code: task_manual_iteration_tutorial.py <task_manual_iteration_tutorial.py>`{.interpreted-text
role="download"}
:::

::: {.container .sphx-glr-download .sphx-glr-download-jupyter}
`Download Jupyter notebook: task_manual_iteration_tutorial.ipynb <task_manual_iteration_tutorial.ipynb>`{.interpreted-text
role="download"}
:::
:::::
::::::
:::::::

:::: only
html

::: rst-class
sphx-glr-signature

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)
:::
::::
