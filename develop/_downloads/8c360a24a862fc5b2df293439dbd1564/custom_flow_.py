"""
================================
Creating and Using a Custom Flow
================================

The most convenient way to create a flow for your machine learning workflow is to generate it
automatically as described in the :ref:`sphx_glr_examples_30_extended_flow_id_tutorial.py` tutorial.
However, there are scenarios where this is not possible, such
as when the flow uses a framework without an extension or when the flow is described by a script.

In those cases you can still create a custom flow by following the steps of this tutorial.
As an example we will use the flows generated for the `AutoML Benchmark <https://openml.github.io/automlbenchmark/>`_,
and also show how to link runs to the custom flow.
"""

# License: BSD 3-Clause

from collections import OrderedDict
import numpy as np

import openml
from openml import OpenMLClassificationTask
from openml.runs.functions import format_prediction

####################################################################################################
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt
openml.config.start_using_configuration_for_example()

####################################################################################################
# 1. Defining the flow
# ====================
# The first step is to define all the hyperparameters of your flow.
# The API pages feature a descriptions of each variable of the :class:`openml.flows.OpenMLFlow`.
# Note that `external version` and `name` together uniquely identify a flow.
#
# The AutoML Benchmark runs AutoML systems across a range of tasks.
# OpenML stores Flows for each AutoML system. However, the AutoML benchmark adds
# preprocessing to the flow, so should be described in a new flow.
#
# We will break down the flow arguments into several groups, for the tutorial.
# First we will define the name and version information.
# Make sure to leave enough information so others can determine exactly which
# version of the package/script is used. Use tags so users can find your flow easily.

general = dict(
    name="automlbenchmark_autosklearn",
    description=(
        "Auto-sklearn as set up by the AutoML Benchmark"
        "Source: https://github.com/openml/automlbenchmark/releases/tag/v0.9"
    ),
    external_version="amlb==0.9",
    language="English",
    tags=["amlb", "benchmark", "study_218"],
    dependencies="amlb==0.9",
)

####################################################################################################
# Next we define the flow hyperparameters. We define their name and default value in `parameters`,
# and provide meta-data for each hyperparameter through `parameters_meta_info`.
# Note that even though the argument name is `parameters` they describe the hyperparameters.
# The use of ordered dicts is required.

flow_hyperparameters = dict(
    parameters=OrderedDict(time="240", memory="32", cores="8"),
    parameters_meta_info=OrderedDict(
        cores=OrderedDict(description="number of available cores", data_type="int"),
        memory=OrderedDict(description="memory in gigabytes", data_type="int"),
        time=OrderedDict(description="time in minutes", data_type="int"),
    ),
)

####################################################################################################
# It is possible to build a flow which uses other flows.
# For example, the Random Forest Classifier is a flow, but you could also construct a flow
# which uses a Random Forest Classifier in a ML pipeline. When constructing the pipeline flow,
# you can use the Random Forest Classifier flow as a *subflow*. It allows for
# all hyperparameters of the Random Classifier Flow to also be specified in your pipeline flow.
#
# Note: you can currently only specific one subflow as part of the components.
#
# In this example, the auto-sklearn flow is a subflow: the auto-sklearn flow is entirely executed as part of this flow.
# This allows people to specify auto-sklearn hyperparameters used in this flow.
# In general, using a subflow is not required.
#
# Note: flow 9313 is not actually the right flow on the test server,
# but that does not matter for this demonstration.

autosklearn_flow = openml.flows.get_flow(9313)  # auto-sklearn 0.5.1
subflow = dict(
    components=OrderedDict(automl_tool=autosklearn_flow),
    # If you do not want to reference a subflow, you can use the following:
    # components=OrderedDict(),
)

####################################################################################################
# With all parameters of the flow defined, we can now initialize the OpenMLFlow and publish.
# Because we provided all the details already, we do not need to provide a `model` to the flow.
#
# In our case, we don't even have a model. It is possible to have a model but still require
# to follow these steps when the model (python object) does not have an extensions from which
# to automatically extract the hyperparameters.
# So whether you have a model with no extension or no model at all, explicitly set
# the model of the flow to `None`.

autosklearn_amlb_flow = openml.flows.OpenMLFlow(
    **general,
    **flow_hyperparameters,
    **subflow,
    model=None,
)
autosklearn_amlb_flow.publish()
print(f"autosklearn flow created: {autosklearn_amlb_flow.flow_id}")

####################################################################################################
# 2. Using the flow
# ====================
# This Section will show how to upload run data for your custom flow.
# Take care to change the values of parameters as well as the task id,
# to reflect the actual run.
# Task and parameter values in the example are fictional.

flow_id = autosklearn_amlb_flow.flow_id

parameters = [
    OrderedDict([("oml:name", "cores"), ("oml:value", 4), ("oml:component", flow_id)]),
    OrderedDict([("oml:name", "memory"), ("oml:value", 16), ("oml:component", flow_id)]),
    OrderedDict([("oml:name", "time"), ("oml:value", 120), ("oml:component", flow_id)]),
]

task_id = 1200  # Iris Task
task = openml.tasks.get_task(task_id)
dataset_id = task.get_dataset().dataset_id


####################################################################################################
# The last bit of information for the run we need are the predicted values.
# The exact format of the predictions will depend on the task.
#
# The predictions should always be a list of lists, each list should contain:
#
# - the repeat number: for repeated evaluation strategies. (e.g. repeated cross-validation)
# - the fold number: for cross-validation. (what should this be for holdout?)
# - 0: this field is for backward compatibility.
# - index: the row (of the original dataset) for which the prediction was made.
# - p_1, ..., p_c: for each class the predicted probability of the sample
#   belonging to that class. (no elements for regression tasks)
#   Make sure the order of these elements follows the order of `task.class_labels`.
# - the predicted class/value for the sample
# - the true class/value for the sample
#
# When using openml-python extensions (such as through `run_model_on_task`),
# all of this formatting is automatic.
# Unfortunately we can not automate this procedure for custom flows,
# which means a little additional effort is required.
#
# Here we generated some random predictions in place.
# You can ignore this code, or use it to better understand the formatting of the predictions.
#
# Find the repeats/folds for this task:
n_repeats, n_folds, _ = task.get_split_dimensions()
all_test_indices = [
    (repeat, fold, index)
    for repeat in range(n_repeats)
    for fold in range(n_folds)
    for index in task.get_train_test_split_indices(fold, repeat)[1]
]

# random class probabilities (Iris has 150 samples and 3 classes):
r = np.random.rand(150 * n_repeats, 3)
# scale the random values so that the probabilities of each sample sum to 1:
y_proba = r / r.sum(axis=1).reshape(-1, 1)
y_pred = y_proba.argmax(axis=1)

class_map = dict(zip(range(3), task.class_labels))
_, y_true = task.get_X_and_y()
y_true = [class_map[y] for y in y_true]

# We format the predictions with the utility function `format_prediction`.
# It will organize the relevant data in the expected format/order.
predictions = []
for where, y, yp, proba in zip(all_test_indices, y_true, y_pred, y_proba):
    repeat, fold, index = where

    prediction = format_prediction(
        task=task,
        repeat=repeat,
        fold=fold,
        index=index,
        prediction=class_map[yp],
        truth=y,
        proba={c: pb for (c, pb) in zip(task.class_labels, proba)},
    )
    predictions.append(prediction)

####################################################################################################
# Finally we can create the OpenMLRun object and upload.
# We use the argument setup_string because the used flow was a script.

benchmark_command = f"python3 runbenchmark.py auto-sklearn medium -m aws -t 119"
my_run = openml.runs.OpenMLRun(
    task_id=task_id,
    flow_id=flow_id,
    dataset_id=dataset_id,
    parameter_settings=parameters,
    setup_string=benchmark_command,
    data_content=predictions,
    tags=["study_218"],
    description_text="Run generated by the Custom Flow tutorial.",
)
my_run.publish()
print("run created:", my_run.run_id)

openml.config.stop_using_configuration_for_example()
