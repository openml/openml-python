::::: only
html

:::: {.note .sphx-glr-download-link-note}
::: title
Note
:::

`Go to the end <sphx_glr_download_examples_20_basic_simple_flows_and_runs_tutorial.py>`{.interpreted-text
role="ref"} to download the full example code
::::
:::::

::: rst-class
sphx-glr-example-title
:::

# Flows and Runs {#sphx_glr_examples_20_basic_simple_flows_and_runs_tutorial.py}

A simple tutorial on how to train/run a model and how to upload the
results.

``` default
# License: BSD 3-Clause

import openml
from sklearn import ensemble, neighbors
```

:::: warning
::: title
Warning
:::

.. include:: ../../test_server_usage_warning.txt
::::

``` default
openml.config.start_using_configuration_for_example()
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/config.py:184: UserWarning: Switching to the test server https://test.openml.org/api/v1/xml to not upload results to the live server. Using the test server may result in reduced performance of the API!
  warnings.warn(
```
:::

# Train a machine learning model

``` default
# NOTE: We are using dataset 20 from the test server: https://test.openml.org/d/20
dataset = openml.datasets.get_dataset(20)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)
clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
```
:::

<div class="output_subarea output_html rendered_html output_result">
<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=3)</pre></div></div></div></div></div>
</div>
<br />
<br />

# Running a model on a task

``` default
task = openml.tasks.get_task(119)
clf = ensemble.RandomForestClassifier()
run = openml.runs.run_model_on_task(clf, task)
print(run)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
OpenML Run
==========
Uploader Name...................: None
Metric..........................: None
Local Result - Accuracy (+- STD): 0.7510 +- 0.0000
Local Runtime - ms (+- STD).....: 95.9373 +- 0.0000
Run ID..........................: None
Task ID.........................: 119
Task Type.......................: None
Task URL........................: https://test.openml.org/t/119
Flow ID.........................: 4417
Flow Name.......................: sklearn.ensemble._forest.RandomForestClassifier
Flow URL........................: https://test.openml.org/f/4417
Setup ID........................: None
Setup String....................: Python_3.10.12. Sklearn_1.3.0. NumPy_1.25.1. SciPy_1.11.1.
Dataset ID......................: 20
Dataset URL.....................: https://test.openml.org/d/20
```
:::

# Publishing the run

``` default
myrun = run.publish()
print(f"Run was uploaded to {myrun.openml_url}")
print(f"The flow can be found at {myrun.flow.openml_url}")
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
Run was uploaded to https://test.openml.org/r/7325
The flow can be found at https://test.openml.org/f/4417
```
:::

``` default
openml.config.stop_using_configuration_for_example()
```

::: rst-class
sphx-glr-timing

**Total running time of the script:** ( 0 minutes 3.799 seconds)
:::

::::::: {#sphx_glr_download_examples_20_basic_simple_flows_and_runs_tutorial.py}
:::::: only
html

::::: {.container .sphx-glr-footer .sphx-glr-footer-example}
::: {.container .sphx-glr-download .sphx-glr-download-python}
`Download Python source code: simple_flows_and_runs_tutorial.py <simple_flows_and_runs_tutorial.py>`{.interpreted-text
role="download"}
:::

::: {.container .sphx-glr-download .sphx-glr-download-jupyter}
`Download Jupyter notebook: simple_flows_and_runs_tutorial.ipynb <simple_flows_and_runs_tutorial.ipynb>`{.interpreted-text
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
