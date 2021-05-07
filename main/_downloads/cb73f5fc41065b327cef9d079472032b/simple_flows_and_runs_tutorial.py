"""
Flows and Runs
==============

A simple tutorial on how to train/run a model and how to upload the results.
"""

# License: BSD 3-Clause

import openml
from sklearn import ensemble, neighbors

############################################################################
# Train a machine learning model
# ==============================
#
# .. warning:: This example uploads data. For that reason, this example
#   connects to the test server at test.openml.org. This prevents the main
#   server from crowding with example datasets, tasks, runs, and so on.

openml.config.start_using_configuration_for_example()

# NOTE: We are using dataset 20 from the test server: https://test.openml.org/d/20
dataset = openml.datasets.get_dataset(20)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)

############################################################################
# Running a model on a task
# =========================

task = openml.tasks.get_task(119)
clf = ensemble.RandomForestClassifier()
run = openml.runs.run_model_on_task(clf, task)
print(run)

############################################################################
# Publishing the run
# ==================

myrun = run.publish()
print("Run was uploaded to http://test.openml.org/r/" + str(myrun.run_id))
print("The flow can be found at http://test.openml.org/f/" + str(myrun.flow_id))

############################################################################
openml.config.stop_using_configuration_for_example()
