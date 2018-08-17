"""
Flows and Runs
==============

How to train/run a model and how to upload/download the information that follows.
"""

import openml
import pandas as pd
import seaborn as sns
from sklearn import ensemble, neighbors, preprocessing, pipeline, tree

############################################################################
# Train machine learning models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Train a scikit-learn model on the data manually.
dataset = openml.datasets.get_dataset(68)
X, y = dataset.get_data(target=dataset.default_target_attribute)
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

############################################################################
# Share the run on the OpenML server

myrun = run.publish()
# For this tutorial, our configuration publishes to the test server
# as to not pollute the main server.
print("Uploaded to http://test.openml.org/r/" + str(myrun.run_id))

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

run = openml.runs.run_flow_on_task(flow, task)
myrun = run.publish()
print("Uploaded to http://test.openml.org/r/" + str(myrun.run_id))

############################################################################
# Download previous results
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can download all your results anytime, as well as everybody else's.
#
# List runs by uploader, flow, task, tag, id, ...

# Get the list of runs for task 115
myruns = openml.runs.list_runs(task=[115], size=100)

# Download the tasks and plot the scores
scores = []
for id, _ in myruns.items():
    run = openml.runs.get_run(id)
    scores.append({"flow": run.flow_name, "score": run.evaluations['area_under_roc_curve']})

sns.violinplot(x="score", y="flow", data=pd.DataFrame(scores), scale="width", palette="Set3")

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

    try:
        run = openml.runs.run_flow_on_task(flow, task)
        myrun = run.publish()
        print("kNN on %s: http://test.openml.org/r/%d" % (data.name, myrun.run_id))
    except openml.exceptions.PyOpenMLError as err:
        print("OpenML: {0}".format(err))
