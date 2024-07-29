::::: only
html

:::: {.note .sphx-glr-download-link-note}
::: title
Note
:::

`Go to the end <sphx_glr_download_examples_30_extended_run_setup_tutorial.py>`{.interpreted-text
role="ref"} to download the full example code
::::
:::::

::: rst-class
sphx-glr-example-title
:::

# Run Setup {#sphx_glr_examples_30_extended_run_setup_tutorial.py}

By: Jan N. van Rijn

One of the key features of the openml-python library is that is allows
to reinstantiate flows with hyperparameter settings that were uploaded
before. This tutorial uses the concept of setups. Although setups are
not extensively described in the OpenML documentation (because most
users will not directly use them), they form a important concept within
OpenML distinguishing between hyperparameter configurations. A setup is
the combination of a flow with all its hyperparameters set.

A key requirement for reinstantiating a flow is to have the same
scikit-learn version as the flow that was uploaded. However, this
tutorial will upload the flow (that will later be reinstantiated)
itself, so it can be ran with any scikit-learn version that is supported
by this library. In this case, the requirement of the corresponding
scikit-learn versions is automatically met.

In this tutorial we will

:   1)  Create a flow and use it to solve a task;
    2)  Download the flow, reinstantiate the model with same
        hyperparameters, and solve the same task again;
    3)  We will verify that the obtained results are exactly the same.

``` default
# License: BSD 3-Clause

import numpy as np
import openml
from openml.extensions.sklearn import cat, cont

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
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

## 1) Create a flow and use it to solve a task

``` default
# first, let's download the task that we are interested in
task = openml.tasks.get_task(6)


# we will create a fairly complex model, with many preprocessing components and
# many potential hyperparameters. Of course, the model can be as complex and as
# easy as you want it to be


cat_imp = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", sparse=False),
    TruncatedSVD(),
)
cont_imp = SimpleImputer(strategy="median")
ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
model_original = Pipeline(
    steps=[
        ("transform", ct),
        ("estimator", RandomForestClassifier()),
    ]
)

# Let's change some hyperparameters. Of course, in any good application we
# would tune them using, e.g., Random Search or Bayesian Optimization, but for
# the purpose of this tutorial we set them to some specific values that might
# or might not be optimal
hyperparameters_original = {
    "estimator__criterion": "gini",
    "estimator__n_estimators": 50,
    "estimator__max_depth": 10,
    "estimator__min_samples_leaf": 1,
}
model_original.set_params(**hyperparameters_original)

# solve the task and upload the result (this implicitly creates the flow)
run = openml.runs.run_model_on_task(model_original, task, avoid_duplicate_runs=False)
run_original = run.publish()  # this implicitly uploads the flow
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
/openml/venv/lib/python3.10/site-packages/sklearn/preprocessing/_encoders.py:972: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
  warnings.warn(
/code/openml/extensions/sklearn/extension.py:1785: UserWarning: Estimator only predicted for 5/6 classes!
  warnings.warn(message)
/code/openml/tasks/functions.py:372: FutureWarning: Starting from Version 0.15.0 `download_splits` will default to ``False`` instead of ``True`` and be independent from `download_data`. To disable this message until version 0.15 explicitly set `download_splits` to a bool.
  warnings.warn(
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
```
:::

## 2) Download the flow and solve the same task again.

``` default
# obtain setup id (note that the setup id is assigned by the OpenML server -
# therefore it was not yet available in our local copy of the run)
run_downloaded = openml.runs.get_run(run_original.run_id)
setup_id = run_downloaded.setup_id

# after this, we can easily reinstantiate the model
model_duplicate = openml.setups.initialize_model(setup_id)
# it will automatically have all the hyperparameters set

# and run the task again
run_duplicate = openml.runs.run_model_on_task(model_duplicate, task, avoid_duplicate_runs=False)
```

::: rst-class
sphx-glr-script-out

``` none
/code/openml/datasets/functions.py:438: FutureWarning: Starting from Version 0.15 `download_data`, `download_qualities`, and `download_features_meta_data` will all be ``False`` instead of ``True`` by default to enable lazy loading. To disable this message until version 0.15 explicitly set `download_data`, `download_qualities`, and `download_features_meta_data` to a bool while calling `get_dataset`.
  warnings.warn(
/openml/venv/lib/python3.10/site-packages/sklearn/preprocessing/_encoders.py:972: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
  warnings.warn(
/code/openml/extensions/sklearn/extension.py:1785: UserWarning: Estimator only predicted for 5/6 classes!
  warnings.warn(message)
```
:::

## 3) We will verify that the obtained results are exactly the same.

``` default
# the run has stored all predictions in the field data content
np.testing.assert_array_equal(run_original.data_content, run_duplicate.data_content)
```

``` default
openml.config.stop_using_configuration_for_example()
```

::: rst-class
sphx-glr-timing

**Total running time of the script:** ( 0 minutes 6.595 seconds)
:::

::::::: {#sphx_glr_download_examples_30_extended_run_setup_tutorial.py}
:::::: only
html

::::: {.container .sphx-glr-footer .sphx-glr-footer-example}
::: {.container .sphx-glr-download .sphx-glr-download-python}
`Download Python source code: run_setup_tutorial.py <run_setup_tutorial.py>`{.interpreted-text
role="download"}
:::

::: {.container .sphx-glr-download .sphx-glr-download-jupyter}
`Download Jupyter notebook: run_setup_tutorial.ipynb <run_setup_tutorial.ipynb>`{.interpreted-text
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
