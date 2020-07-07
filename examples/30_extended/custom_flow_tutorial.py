"""
================================
Creating and Using a Custom Flow
================================

The most convenient way to create a flow for your machine learning workflow is to generate it
automatically as described in <>. However, there are scenarios where this is not possible, such
as when the flow uses a framework without an extension or when the flow is described by a script.

In those cases you can still create a custom flow by following the steps of this tutorial.
As an example we will use the flows generated for the AutoML Benchmark (...),
and also show how to link runs to the custom flow.
"""

####################################################################################################

# License: BSD 3-Clause
# .. warning:: This example uploads data. For that reason, this example
#   connects to the test server at test.openml.org. This prevents the main
#   server from crowding with example datasets, tasks, runs, and so on.
from collections import OrderedDict

import openml

openml.config.start_using_configuration_for_example()

####################################################################################################
# 1. Defining the flow
# ====================
# The first step is to define all the hyperparameters of your flow.
# Check ... for the descriptions of each variable.
# Note that `external version` and `name` together should uniquely identify a flow.
#
# The AutoML Benchmark runs AutoML systems across a range of tasks.
# We can not use the flows of the AutoML systems directly, as the benchmark adds performs
# preprocessing as required.
#
# We will break down the flow parameters into several groups, for the tutorial.
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
# and provide meta-data for each parameter through `parameters_meta_info`.
# Note that the use of ordered dicts is required.

flow_hyperparameters = dict(
    parameters=OrderedDict(time="240", memory="32", cores="8"),
    parameters_meta_info=OrderedDict(
        cores=OrderedDict(description="number of available cores", data_type="int"),
        memory=OrderedDict(description="memory in gigabytes", data_type="int"),
        time=OrderedDict(description="time in minutes", data_type="int"),
    ),
)

####################################################################################################
# It is possible for flows to contain subflows. In this example, the auto-sklearn flow is a
# subflow, this means that the subflow is entirely executed as part of this flow.
# Using this modularity also allows your runs to specify which hyperparameters of the
# subflows were used!
#
# Note: flow 15275 is not actually the right flow on the test server,
# but that does not matter for this demonstration.

autosklearn_flow = openml.flows.get_flow(15275)  # auto-sklearn 0.5.1
subflow = dict(components=OrderedDict(automl_tool=autosklearn_flow),)

####################################################################################################
# With all parameters of the flow defined, we can now initialize the OpenMLFlow and publish.
# Explicitly set the model of the flow to `None`, because we provided all the details already!

autosklearn_amlb_flow = openml.flows.OpenMLFlow(
    **general, **flow_hyperparameters, **subflow, model=None,
)
autosklearn_amlb_flow.publish()
print(f"autosklearn flow created: {autosklearn_amlb_flow.flow_id}")
# for dev purposes, since we're rerunning this often, we want to double-check no new flows are created

####################################################################################################
# 2. Using the flow
# ====================
# This Section will show how to upload run data for your custom flow.
#
