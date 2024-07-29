::::: only
html

:::: {.note .sphx-glr-download-link-note}
::: title
Note
:::

`Go to the end <sphx_glr_download_examples_30_extended_fetch_runtimes_tutorial.py>`{.interpreted-text
role="ref"} to download the full example code
::::
:::::

::: rst-class
sphx-glr-example-title
:::

# Measuring runtimes for Scikit-learn models {#sphx_glr_examples_30_extended_fetch_runtimes_tutorial.py}

The runtime of machine learning models on specific datasets can be a
deciding factor on the choice of algorithms, especially for benchmarking
and comparison purposes. OpenML\'s scikit-learn extension provides
runtime data from runs of model fit and prediction on tasks or datasets,
for both the CPU-clock as well as the actual wallclock-time incurred.
The objective of this example is to illustrate how to retrieve such
timing measures, and also offer some potential means of usage and
interpretation of the same.

It should be noted that there are multiple levels at which parallelism
can occur.

-   At the outermost level, OpenML tasks contain fixed data splits, on
    which the defined model/flow is executed. Thus, a model can be fit
    on each OpenML dataset fold in parallel using the
    [n_jobs]{.title-ref} parameter to [run_model_on_task]{.title-ref} or
    [run_flow_on_task]{.title-ref} (illustrated under Case 2 & 3 below).
-   The model/flow specified can also include scikit-learn models that
    perform their own parallelization. For instance, by specifying
    [n_jobs]{.title-ref} in a Random Forest model definition (covered
    under Case 2 below).
-   The sklearn model can further be an HPO estimator and contain it\'s
    own parallelization. If the base estimator used also supports
    [parallelization]{.title-ref}, then there\'s at least a 2-level
    nested definition for parallelization possible (covered under Case 3
    below).

We shall cover these 5 representative scenarios for:

-   (Case 1) Retrieving runtimes for Random Forest training and
    prediction on each of the cross-validation folds
-   (Case 2) Testing the above setting in a parallel setup and monitor
    the difference using runtimes retrieved
-   (Case 3) Comparing RandomSearchCV and GridSearchCV on the above task
    based on runtimes
-   (Case 4) Running models that don\'t run in parallel or models which
    scikit-learn doesn\'t parallelize
-   (Case 5) Running models that do not release the Python Global
    Interpreter Lock (GIL)

``` default
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
```

## Preparing tasks and scikit-learn models

``` default
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
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
OpenML Classification Task
==========================
Task Type Description: https://www.openml.org/tt/TaskType.SUPERVISED_CLASSIFICATION
Task ID..............: 167119
Task URL.............: https://www.openml.org/t/167119
Estimation Procedure.: crossvalidation
Target Feature.......: class
# of Classes.........: 3
Cost Matrix..........: Available
Task 167119: number of repeats: 1, number of folds: 10, number of samples 1.
```
:::

## Case 1: Running a Random Forest model on an OpenML task

We\'ll run a Random Forest model and obtain an OpenML run object. We can
see the evaluations recorded per fold for the dataset and the
information available for this run.

``` default
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
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
The timing and performance metrics available: 
usercpu_time_millis_training
wall_clock_time_millis_training
usercpu_time_millis_testing
usercpu_time_millis
wall_clock_time_millis_testing
wall_clock_time_millis
predictive_accuracy

The performance metric is recorded under `predictive_accuracy` per fold and can be retrieved as: 
Repeat #0-Fold #0: 0.7834
Repeat #0-Fold #1: 0.7773
Repeat #0-Fold #2: 0.7813
Repeat #0-Fold #3: 0.7811
Repeat #0-Fold #4: 0.7863
Repeat #0-Fold #5: 0.7829
Repeat #0-Fold #6: 0.7718
Repeat #0-Fold #7: 0.7798
Repeat #0-Fold #8: 0.7825
Repeat #0-Fold #9: 0.7735
```
:::

The remaining entries recorded in [measures]{.title-ref} are the runtime
records related as:

usercpu_time_millis = usercpu_time_millis_training +
usercpu_time_millis_testing

wall_clock_time_millis = wall_clock_time_millis_training +
wall_clock_time_millis_testing

The timing measures recorded as [\*\_millis_training]{.title-ref}
contain the per repeat-per fold timing incurred for the execution of the
[.fit()]{.title-ref} procedure of the model. For
[usercpu_time\_\*]{.title-ref} the time recorded using
[time.process_time()]{.title-ref} is converted to
[milliseconds]{.title-ref} and stored. Similarly,
[time.time()]{.title-ref} is used to record the time entry for
[wall_clock_time\_\*]{.title-ref}. The [\*\_millis_testing]{.title-ref}
entry follows the same procedure but for time taken for the
[.predict()]{.title-ref} procedure.

``` default
# Comparing the CPU and wall-clock training times of the Random Forest model
print_compare_runtimes(measures)
```

::: rst-class
sphx-glr-script-out

``` none
Repeat #0-Fold #0: CPU-235.799 vs Wall-235.806
Repeat #0-Fold #1: CPU-234.915 vs Wall-234.942
Repeat #0-Fold #2: CPU-215.000 vs Wall-215.677
Repeat #0-Fold #3: CPU-218.780 vs Wall-218.210
Repeat #0-Fold #4: CPU-210.419 vs Wall-210.423
Repeat #0-Fold #5: CPU-210.813 vs Wall-211.084
Repeat #0-Fold #6: CPU-210.696 vs Wall-218.624
Repeat #0-Fold #7: CPU-206.779 vs Wall-206.782
Repeat #0-Fold #8: CPU-219.276 vs Wall-218.185
Repeat #0-Fold #9: CPU-211.948 vs Wall-211.660
```
:::

## Case 2: Running Scikit-learn model on an OpenML task in parallel

Redefining the model to allow parallelism with [n_jobs=2]{.title-ref} (2
cores)

``` default
clf = RandomForestClassifier(n_estimators=10, n_jobs=2)

run2 = openml.runs.run_model_on_task(
    model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False
)
measures = run2.fold_evaluations
# The wall-clock time recorded per fold should be lesser than Case 1 above
print_compare_runtimes(measures)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Repeat #0-Fold #0: CPU-291.231 vs Wall-177.857
Repeat #0-Fold #1: CPU-247.121 vs Wall-160.623
Repeat #0-Fold #2: CPU-299.050 vs Wall-171.314
Repeat #0-Fold #3: CPU-219.565 vs Wall-130.540
Repeat #0-Fold #4: CPU-214.352 vs Wall-128.408
Repeat #0-Fold #5: CPU-214.530 vs Wall-129.069
Repeat #0-Fold #6: CPU-216.249 vs Wall-131.271
Repeat #0-Fold #7: CPU-214.707 vs Wall-129.280
Repeat #0-Fold #8: CPU-215.784 vs Wall-130.975
Repeat #0-Fold #9: CPU-224.041 vs Wall-127.277
```
:::

Running a Random Forest model on an OpenML task in parallel (all cores
available):

``` default
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
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Repeat #0-Fold #0: CPU-223.135 vs Wall-141.702
Repeat #0-Fold #1: CPU-252.852 vs Wall-157.798
Repeat #0-Fold #2: CPU-225.624 vs Wall-144.946
Repeat #0-Fold #3: CPU-215.932 vs Wall-137.153
Repeat #0-Fold #4: CPU-218.327 vs Wall-132.138
Repeat #0-Fold #5: CPU-216.836 vs Wall-128.753
Repeat #0-Fold #6: CPU-217.526 vs Wall-141.789
Repeat #0-Fold #7: CPU-218.256 vs Wall-135.525
Repeat #0-Fold #8: CPU-218.904 vs Wall-134.427
Repeat #0-Fold #9: CPU-221.687 vs Wall-130.505
```
:::

We can now observe that the ratio of CPU time to wallclock time is lower
than in case 1. This happens because joblib by default spawns
subprocesses for the workloads for which CPU time cannot be tracked.
Therefore, interpreting the reported CPU and wallclock time requires
knowledge of the parallelization applied at runtime.

Running the same task with a different parallel backend. Joblib provides
multiple backends: {[loky]{.title-ref} (default),
[multiprocessing]{.title-ref}, [dask]{.title-ref},
[threading]{.title-ref}, [sequential]{.title-ref}}. The backend can be
explicitly set using a joblib context manager. The behaviour of the job
distribution can change and therefore the scale of runtimes recorded
too.

``` default
with parallel_backend(backend="multiprocessing", n_jobs=-1):
    run3_ = openml.runs.run_model_on_task(
        model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False
    )
measures = run3_.fold_evaluations
print_compare_runtimes(measures)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Repeat #0-Fold #0: CPU-324.893 vs Wall-545.866
Repeat #0-Fold #1: CPU-248.720 vs Wall-412.322
Repeat #0-Fold #2: CPU-269.108 vs Wall-424.729
Repeat #0-Fold #3: CPU-264.310 vs Wall-401.721
Repeat #0-Fold #4: CPU-262.941 vs Wall-379.026
Repeat #0-Fold #5: CPU-242.700 vs Wall-376.538
Repeat #0-Fold #6: CPU-257.596 vs Wall-418.498
Repeat #0-Fold #7: CPU-233.435 vs Wall-437.289
Repeat #0-Fold #8: CPU-257.681 vs Wall-409.761
Repeat #0-Fold #9: CPU-282.320 vs Wall-428.106
```
:::

The CPU time interpretation becomes ambiguous when jobs are distributed
over an unknown number of cores or when subprocesses are spawned for
which the CPU time cannot be tracked, as in the examples above. It is
impossible for OpenML-Python to capture the availability of the number
of cores/threads, their eventual utilisation and whether workloads are
executed in subprocesses, for various cases that can arise as
demonstrated in the rest of the example. Therefore, the final
interpretation of the runtimes is left to the [user]{.title-ref}.

## Case 3: Running and benchmarking HPO algorithms with their runtimes

We shall now optimize a similar RandomForest model for the same task
using scikit-learn\'s HPO support by using GridSearchCV to optimize our
earlier RandomForest model\'s hyperparameter [n_estimators]{.title-ref}.
Scikit-learn also provides a [refit_time\_]{.title-ref} for such HPO
models, i.e., the time incurred by training and evaluating the model on
the best found parameter setting. This is included in the
[wall_clock_time_millis_training]{.title-ref} measure recorded.

``` default
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
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Repeat #0-Fold #0: CPU-4216.657 vs Wall-3927.152
Repeat #0-Fold #1: CPU-4267.256 vs Wall-3532.315
Repeat #0-Fold #2: CPU-4153.419 vs Wall-3756.332
Repeat #0-Fold #3: CPU-4165.637 vs Wall-3484.633
Repeat #0-Fold #4: CPU-4148.764 vs Wall-3386.673
Repeat #0-Fold #5: CPU-4409.518 vs Wall-3800.539
Repeat #0-Fold #6: CPU-4381.566 vs Wall-3520.530
Repeat #0-Fold #7: CPU-4377.360 vs Wall-3966.456
Repeat #0-Fold #8: CPU-4427.580 vs Wall-3939.647
Repeat #0-Fold #9: CPU-4459.380 vs Wall-3545.670
```
:::

Like any optimisation problem, scikit-learn\'s HPO estimators also
generate a sequence of configurations which are evaluated, using which
the best found configuration is tracked throughout the trace. The OpenML
run object stores these traces as OpenMLRunTrace objects accessible
using keys of the pattern (repeat, fold, iterations). Here
[fold]{.title-ref} implies the outer-cross validation fold as obtained
from the task data splits in OpenML. GridSearchCV here performs grid
search over the inner-cross validation folds as parameterized by the
[cv]{.title-ref} parameter. Since [GridSearchCV]{.title-ref} in this
example performs a [2-fold]{.title-ref} cross validation, the runtime
recorded per repeat-per fold in the run object is for the entire
[fit()]{.title-ref} procedure of GridSearchCV thus subsuming the
runtimes of the 2-fold (inner) CV search performed.

``` default
# We earlier extracted the number of repeats and folds for this task:
print("# repeats: {}\n# folds: {}".format(n_repeats, n_folds))

# To extract the training runtime of the first repeat, first fold:
print(run4.fold_evaluations["wall_clock_time_millis_training"][0][0])
```

::: rst-class
sphx-glr-script-out

``` none
# repeats: 1
# folds: 10
3927.151918411255
```
:::

To extract the training runtime of the 1-st repeat, 4-th (outer) fold
and also to fetch the parameters and performance of the evaluations made
during the 1-st repeat, 4-th fold evaluation by the Grid Search model.

``` default
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
```

::: rst-class
sphx-glr-script-out

``` none
Total runtime for repeat 0's fold 3: 3484.633446 ms
n_estimators:  1 - score: 0.759
n_estimators: 13 - score: 0.799
n_estimators: 25 - score: 0.803
n_estimators: 37 - score: 0.804
n_estimators: 50 - score: 0.803
```
:::

Scikit-learn\'s HPO estimators also come with an argument
[refit=True]{.title-ref} as a default. In our previous model definition
it was set to True by default, which meant that the best found
hyperparameter configuration was used to refit or retrain the model
without any inner cross validation. This extra refit time measure is
provided by the scikit-learn model as the attribute
[refit_time\_]{.title-ref}. This time is included in the
[wall_clock_time_millis_training]{.title-ref} measure.

For non-HPO estimators, [wall_clock_time_millis =
wall_clock_time_millis_training +
wall_clock_time_millis_testing]{.title-ref}.

For HPO estimators, [wall_clock_time_millis =
wall_clock_time_millis_training + wall_clock_time_millis_testing +
refit_time]{.title-ref}.

This refit time can therefore be explicitly extracted in this manner:

``` default
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
```

::: rst-class
sphx-glr-script-out

``` none
Repeat #0-Fold #0: 665.5693
Repeat #0-Fold #1: 539.9771
Repeat #0-Fold #2: 706.4621
Repeat #0-Fold #3: 634.3455
Repeat #0-Fold #4: 598.8989
Repeat #0-Fold #5: 755.9273
Repeat #0-Fold #6: 806.9417
Repeat #0-Fold #7: 764.6010
Repeat #0-Fold #8: 825.0468
Repeat #0-Fold #9: 550.1802
```
:::

Along with the GridSearchCV already used above, we demonstrate how such
optimisation traces can be retrieved by showing an application of these
traces - comparing the speed of finding the best configuration using
RandomizedSearchCV and GridSearchCV available with scikit-learn.

``` default
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
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
```
:::

Since for the call to `openml.runs.run_model_on_task` the parameter
`n_jobs` is set to its default `None`, the evaluations across the OpenML
folds are not parallelized. Hence, the time recorded is agnostic to the
`n_jobs` being set at both the HPO estimator `GridSearchCV` as well as
the base estimator `RandomForestClassifier` in this case. The OpenML
extension only records the time taken for the completion of the complete
`fit()` call, per-repeat per-fold.

This notion can be used to extract and plot the best found performance
per fold by the HPO model and the corresponding time taken for search
across that fold. Moreover, since `n_jobs=None` for
`openml.runs.run_model_on_task` the runtimes per fold can be
cumulatively added to plot the trace against time.

``` default
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
```

::: {.image-sg .sphx-glr-single-img alt="Optimisation Trace Comparison" srcset="/examples/30_extended/images/sphx_glr_fetch_runtimes_tutorial_001.png"}
/examples/30_extended/images/sphx_glr_fetch_runtimes_tutorial_001.png
:::

## Case 4: Running models that scikit-learn doesn\'t parallelize

Both scikit-learn and OpenML depend on parallelism implemented through
[joblib]{.title-ref}. However, there can be cases where either models
cannot be parallelized or don\'t depend on joblib for its parallelism. 2
such cases are illustrated below.

Running a Decision Tree model that doesn\'t support parallelism
implicitly, but using OpenML to parallelize evaluations for the
outer-cross validation folds.

``` default
dt = DecisionTreeClassifier()

run6 = openml.runs.run_model_on_task(
    model=dt, task=task, upload_flow=False, avoid_duplicate_runs=False, n_jobs=2
)
measures = run6.fold_evaluations
print_compare_runtimes(measures)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Repeat #0-Fold #0: CPU-73.760 vs Wall-73.762
Repeat #0-Fold #1: CPU-72.873 vs Wall-72.874
Repeat #0-Fold #2: CPU-72.735 vs Wall-72.745
Repeat #0-Fold #3: CPU-74.092 vs Wall-74.092
Repeat #0-Fold #4: CPU-73.657 vs Wall-73.658
Repeat #0-Fold #5: CPU-72.254 vs Wall-72.257
Repeat #0-Fold #6: CPU-73.470 vs Wall-73.470
Repeat #0-Fold #7: CPU-72.530 vs Wall-72.534
Repeat #0-Fold #8: CPU-74.001 vs Wall-74.018
Repeat #0-Fold #9: CPU-73.600 vs Wall-73.602
```
:::

Although the decision tree does not run in parallel, it can release the
[Python
GIL](https://docs.python.org/dev/glossary.html#term-global-interpreter-lock).
This can result in surprising runtime measures as demonstrated below:

``` default
with parallel_backend("threading", n_jobs=-1):
    run7 = openml.runs.run_model_on_task(
        model=dt, task=task, upload_flow=False, avoid_duplicate_runs=False
    )
measures = run7.fold_evaluations
print_compare_runtimes(measures)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Repeat #0-Fold #0: CPU-80.470 vs Wall-80.463
Repeat #0-Fold #1: CPU-74.982 vs Wall-74.985
Repeat #0-Fold #2: CPU-83.721 vs Wall-83.721
Repeat #0-Fold #3: CPU-88.097 vs Wall-88.118
Repeat #0-Fold #4: CPU-77.615 vs Wall-77.624
Repeat #0-Fold #5: CPU-95.236 vs Wall-95.238
Repeat #0-Fold #6: CPU-98.978 vs Wall-100.188
Repeat #0-Fold #7: CPU-83.243 vs Wall-83.185
Repeat #0-Fold #8: CPU-89.183 vs Wall-89.200
Repeat #0-Fold #9: CPU-97.797 vs Wall-95.633
```
:::

Running a Neural Network from scikit-learn that uses scikit-learn
independent parallelism using libraries such as [MKL, OpenBLAS or
BLIS](https://scikit-learn.org/stable/computing/parallelism.html#parallel-numpy-and-scipy-routines-from-numerical-libraries).

``` default
mlp = MLPClassifier(max_iter=10)

run8 = openml.runs.run_model_on_task(
    model=mlp, task=task, upload_flow=False, avoid_duplicate_runs=False
)
measures = run8.fold_evaluations
print_compare_runtimes(measures)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
  warnings.warn(
Repeat #0-Fold #0: CPU-866.453 vs Wall-866.533
Repeat #0-Fold #1: CPU-918.903 vs Wall-756.131
Repeat #0-Fold #2: CPU-997.260 vs Wall-727.315
Repeat #0-Fold #3: CPU-859.797 vs Wall-657.584
Repeat #0-Fold #4: CPU-922.494 vs Wall-701.211
Repeat #0-Fold #5: CPU-1112.121 vs Wall-817.650
Repeat #0-Fold #6: CPU-908.489 vs Wall-682.244
Repeat #0-Fold #7: CPU-889.331 vs Wall-624.544
Repeat #0-Fold #8: CPU-908.252 vs Wall-630.604
Repeat #0-Fold #9: CPU-888.762 vs Wall-654.652
```
:::

## Case 5: Running Scikit-learn models that don\'t release GIL

Certain Scikit-learn models do not release the [Python
GIL](https://docs.python.org/dev/glossary.html#term-global-interpreter-lock)
and are also not executed in parallel via a BLAS library. In such cases,
the CPU times and wallclock times are most likely trustworthy. Note
however that only very few models such as naive Bayes models are of this
kind.

``` default
clf = GaussianNB()

with parallel_backend("multiprocessing", n_jobs=-1):
    run9 = openml.runs.run_model_on_task(
        model=clf, task=task, upload_flow=False, avoid_duplicate_runs=False
    )
measures = run9.fold_evaluations
print_compare_runtimes(measures)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Repeat #0-Fold #0: CPU-26.803 vs Wall-26.800
Repeat #0-Fold #1: CPU-26.351 vs Wall-26.349
Repeat #0-Fold #2: CPU-26.371 vs Wall-26.370
Repeat #0-Fold #3: CPU-26.402 vs Wall-26.400
Repeat #0-Fold #4: CPU-26.782 vs Wall-26.781
Repeat #0-Fold #5: CPU-26.479 vs Wall-26.479
Repeat #0-Fold #6: CPU-26.393 vs Wall-26.390
Repeat #0-Fold #7: CPU-26.430 vs Wall-26.427
Repeat #0-Fold #8: CPU-26.654 vs Wall-26.657
Repeat #0-Fold #9: CPU-26.409 vs Wall-26.406
```
:::

## Summmary

The scikit-learn extension for OpenML-Python records model runtimes for
the CPU-clock and the wall-clock times. The above examples illustrated
how these recorded runtimes can be extracted when using a scikit-learn
model and under parallel setups too. To summarize, the scikit-learn
extension measures the:

-   [CPU-time]{.title-ref} & [wallclock-time]{.title-ref} for the whole
    run
    -   A run here corresponds to a call to
        [run_model_on_task]{.title-ref} or
        [run_flow_on_task]{.title-ref}
    -   The recorded time is for the model fit for each of the
        outer-cross validations folds, i.e., the OpenML data splits
-   Python\'s [time]{.title-ref} module is used to compute the runtimes
    -   [CPU-time]{.title-ref} is recorded using the responses of
        [time.process_time()]{.title-ref}
    -   [wallclock-time]{.title-ref} is recorded using the responses of
        [time.time()]{.title-ref}
-   The timings recorded by OpenML per outer-cross validation fold is
    agnostic to model parallelisation
    -   The wallclock times reported in Case 2 above highlights the
        speed-up on using [n_jobs=-1]{.title-ref} in comparison to
        [n_jobs=2]{.title-ref}, since the timing recorded by OpenML is
        for the entire [fit()]{.title-ref} procedure, whereas the
        parallelisation is performed inside [fit()]{.title-ref} by
        scikit-learn
    -   The CPU-time for models that are run in parallel can be
        difficult to interpret
-   [CPU-time]{.title-ref} & [wallclock-time]{.title-ref} for each
    search per outer fold in an HPO run
    -   Reports the total time for performing search on each of the
        OpenML data split, subsuming any sort of parallelism that
        happened as part of the HPO estimator or the underlying base
        estimator
    -   Also allows extraction of the [refit_time]{.title-ref} that
        scikit-learn measures using [time.time()]{.title-ref} for
        retraining the model per outer fold, for the best found
        configuration
-   [CPU-time]{.title-ref} & [wallclock-time]{.title-ref} for models
    that scikit-learn doesn\'t parallelize
    -   Models like Decision Trees or naive Bayes don\'t parallelize and
        thus both the wallclock and CPU times are similar in runtime for
        the OpenML call
    -   However, models implemented in Cython, such as the Decision
        Trees can release the GIL and still run in parallel if a
        [threading]{.title-ref} backend is used by joblib.
    -   Scikit-learn Neural Networks can undergo parallelization
        implicitly owing to thread-level parallelism involved in the
        linear algebraic operations and thus the wallclock-time and
        CPU-time can differ.

Because of all the cases mentioned above it is crucial to understand
which case is triggered when reporting runtimes for scikit-learn models
measured with OpenML-Python!

::: rst-class
sphx-glr-timing

**Total running time of the script:** ( 1 minutes 13.261 seconds)
:::

::::::: {#sphx_glr_download_examples_30_extended_fetch_runtimes_tutorial.py}
:::::: only
html

::::: {.container .sphx-glr-footer .sphx-glr-footer-example}
::: {.container .sphx-glr-download .sphx-glr-download-python}
`Download Python source code: fetch_runtimes_tutorial.py <fetch_runtimes_tutorial.py>`{.interpreted-text
role="download"}
:::

::: {.container .sphx-glr-download .sphx-glr-download-jupyter}
`Download Jupyter notebook: fetch_runtimes_tutorial.ipynb <fetch_runtimes_tutorial.ipynb>`{.interpreted-text
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
