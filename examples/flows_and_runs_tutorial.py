"""
Flows and Runs
==============

How to train/run a model and how to upload the results.
"""

import openml
import pandas as pd
import seaborn as sns
from pprint import pprint
from sklearn import ensemble, neighbors, preprocessing, pipeline, tree

############################################################################
# Train machine learning models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Train a scikit-learn model on the data manually.

dataset = openml.datasets.get_dataset(68)
X, y = dataset.get_data(
    target=dataset.default_target_attribute
)
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

############################################################################
# You can also ask for meta-data to automatically preprocess the data.
#
# * e.g. categorical features -> do feature encoding
dataset = openml.datasets.get_dataset(17)
X, y, categorical = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True,
)
print("Categorical features: %s" % categorical)
enc = preprocessing.OneHotEncoder(categorical_features=categorical)
X = enc.fit_transform(X)
clf.fit(X, y)

############################################################################
# Runs: Easily explore models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can run (many) scikit-learn algorithms on (many) OpenML tasks.

# Get a task
task = openml.tasks.get_task(403)

# Build any classifier or pipeline
clf = tree.ExtraTreeClassifier()

# Create a flow
flow = openml.flows.sklearn_to_flow(clf)

# Run the flow
run = openml.runs.run_flow_on_task(flow, task)

# pprint(vars(run), depth=2)

############################################################################
# Share the run on the OpenML server
#
# So far the run is only available locally. By calling the publish function, the run is send to the OpenML server:

myrun = run.publish()
# For this tutorial, our configuration publishes to the test server
# as to not pollute the main server.
print("Uploaded to http://test.openml.org/r/" + str(myrun.run_id))

############################################################################
# We can now also inspect the flow object which was automatically created:

flow = openml.flows.get_flow(run.flow_id)
pprint(vars(flow), depth=1)

############################################################################
# It also works with pipelines
# ############################
#
# When you need to handle 'dirty' data, build pipelines to model then automatically.
task = openml.tasks.get_task(115)
pipe = pipeline.Pipeline(steps=[
    ('Imputer', preprocessing.Imputer(strategy='median')),
    ('OneHotEncoder', preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')),
    ('Classifier', ensemble.RandomForestClassifier())
])
flow = openml.flows.sklearn_to_flow(pipe)

run = openml.runs.run_flow_on_task(flow, task, avoid_duplicate_runs=False)
myrun = run.publish()
print("Uploaded to http://test.openml.org/r/" + str(myrun.run_id))

############################################################################
# Challenge
# ^^^^^^^^^
#
# Try to build the best possible models on several OpenML tasks,
# compare your results with the rest of the class and learn from
# them. Some tasks you could try (or browse openml.org):
#
# * EEG eye state: data_id:`1471 <http://www.openml.org/d/1471>`_, task_id:`14951 <http://www.openml.org/t/14951>`_
# * Volcanoes on Venus: data_id:`1527 <http://www.openml.org/d/1527>`_, task_id:`10103 <http://www.openml.org/t/10103>`_
# * Walking activity: data_id:`1509 <http://www.openml.org/d/1509>`_, task_id:`9945 <http://www.openml.org/t/9945>`_, 150k instances.
# * Covertype (Satellite): data_id:`150 <http://www.openml.org/d/150>`_, task_id:`218 <http://www.openml.org/t/218>`_, 500k instances.
# * Higgs (Physics): data_id:`23512 <http://www.openml.org/d/23512>`_, task_id:`52950 <http://www.openml.org/t/52950>`_, 100k instances, missing values.

# Easy benchmarking:
for task_id in [115, ]:  # Add further tasks. Disclaimer: they might take some time
    task = openml.tasks.get_task(task_id)
    data = openml.datasets.get_dataset(task.dataset_id)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    flow = openml.flows.sklearn_to_flow(clf)

    run = openml.runs.run_flow_on_task(flow, task, avoid_duplicate_runs=False)
    myrun = run.publish()
    print("kNN on %s: http://test.openml.org/r/%d" % (data.name, myrun.run_id))
