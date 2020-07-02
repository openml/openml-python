"""
=========
Run Setup
=========

By: Jan N. van Rijn

One of the key features of the openml-python library is that is allows to
reinstantiate flows with hyperparameter settings that were uploaded before.
This tutorial uses the concept of setups. Although setups are not extensively
described in the OpenML documentation (because most users will not directly
use them), they form a important concept within OpenML distinguishing between
hyperparameter configurations.
A setup is the combination of a flow with all its hyperparameters set.

A key requirement for reinstantiating a flow is to have the same scikit-learn
version as the flow that was uploaded. However, this tutorial will upload the
flow (that will later be reinstantiated) itself, so it can be ran with any
scikit-learn version that is supported by this library. In this case, the
requirement of the corresponding scikit-learn versions is automatically met.

In this tutorial we will
    1) Create a flow and use it to solve a task;
    2) Download the flow, reinstantiate the model with same hyperparameters,
       and solve the same task again;
    3) We will verify that the obtained results are exactly the same.

.. warning:: This example uploads data. For that reason, this example
   connects to the test server at test.openml.org. This prevents the main
   server from crowding with example datasets, tasks, runs, and so on.
"""

# License: BSD 3-Clause

import numpy as np
import openml
import sklearn.ensemble
import sklearn.impute
import sklearn.preprocessing


openml.config.start_using_configuration_for_example()

###############################################################################
# 1) Create a flow and use it to solve a task
###############################################################################

# first, let's download the task that we are interested in
task = openml.tasks.get_task(6)


# we will create a fairly complex model, with many preprocessing components and
# many potential hyperparameters. Of course, the model can be as complex and as
# easy as you want it to be
model_original = sklearn.pipeline.make_pipeline(
    sklearn.impute.SimpleImputer(), sklearn.ensemble.RandomForestClassifier()
)


# Let's change some hyperparameters. Of course, in any good application we
# would tune them using, e.g., Random Search or Bayesian Optimization, but for
# the purpose of this tutorial we set them to some specific values that might
# or might not be optimal
hyperparameters_original = {
    "simpleimputer__strategy": "median",
    "randomforestclassifier__criterion": "entropy",
    "randomforestclassifier__max_features": 0.2,
    "randomforestclassifier__min_samples_leaf": 1,
    "randomforestclassifier__n_estimators": 16,
    "randomforestclassifier__random_state": 42,
}
model_original.set_params(**hyperparameters_original)

# solve the task and upload the result (this implicitly creates the flow)
run = openml.runs.run_model_on_task(model_original, task, avoid_duplicate_runs=False)
run_original = run.publish()  # this implicitly uploads the flow

###############################################################################
# 2) Download the flow and solve the same task again.
###############################################################################

# obtain setup id (note that the setup id is assigned by the OpenML server -
# therefore it was not yet available in our local copy of the run)
run_downloaded = openml.runs.get_run(run_original.run_id)
setup_id = run_downloaded.setup_id

# after this, we can easily reinstantiate the model
model_duplicate = openml.setups.initialize_model(setup_id)
# it will automatically have all the hyperparameters set

# and run the task again
run_duplicate = openml.runs.run_model_on_task(model_duplicate, task, avoid_duplicate_runs=False)


###############################################################################
# 3) We will verify that the obtained results are exactly the same.
###############################################################################

# the run has stored all predictions in the field data content
np.testing.assert_array_equal(run_original.data_content, run_duplicate.data_content)

###############################################################################

openml.config.stop_using_configuration_for_example()
