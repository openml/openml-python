"""
Flows and Runs
==============

How to train/run a model and how to upload the results.
"""

# License: BSD 3-Clause

import openml
from sklearn import compose, ensemble, impute, neighbors, preprocessing, pipeline, tree


############################################################################
# We'll use the test server for the rest of this tutorial.
#
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt
openml.config.start_using_configuration_for_example()

############################################################################
# Train machine learning models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Train a scikit-learn model on the data manually.

# NOTE: We are using dataset 68 from the test server: https://test.openml.org/d/68
dataset = openml.datasets.get_dataset(68)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

############################################################################
# You can also ask for meta-data to automatically preprocess the data.
#
# * e.g. categorical features -> do feature encoding
dataset = openml.datasets.get_dataset(17)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)
print(f"Categorical features: {categorical_indicator}")
transformer = compose.ColumnTransformer(
    [("one_hot_encoder", preprocessing.OneHotEncoder(categories="auto"), categorical_indicator)]
)
X = transformer.fit_transform(X)
clf.fit(X, y)

############################################################################
# Runs: Easily explore models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can run (many) scikit-learn algorithms on (many) OpenML tasks.

# Get a task
task = openml.tasks.get_task(403)

# Build any classifier or pipeline
clf = tree.DecisionTreeClassifier()

# Run the flow
run = openml.runs.run_model_on_task(clf, task)

print(run)

############################################################################
# Share the run on the OpenML server
#
# So far the run is only available locally. By calling the publish function,
# the run is sent to the OpenML server:

myrun = run.publish()
# For this tutorial, our configuration publishes to the test server
# as to not pollute the main server.
print(f"Uploaded to {myrun.openml_url}")

############################################################################
# We can now also inspect the flow object which was automatically created:

flow = openml.flows.get_flow(run.flow_id)
print(flow)

############################################################################
# It also works with pipelines
# ############################
#
# When you need to handle 'dirty' data, build pipelines to model then automatically.
# To demonstrate this using the dataset `credit-a <https://test.openml.org/d/16>`_ via
# `task <https://test.openml.org/t/96>`_ as it contains both numerical and categorical
# variables and missing values in both.
task = openml.tasks.get_task(96)

# OpenML helper functions for sklearn can be plugged in directly for complicated pipelines
from openml.extensions.sklearn import cat, cont

pipe = pipeline.Pipeline(
    steps=[
        (
            "Preprocessing",
            compose.ColumnTransformer(
                [
                    (
                        "categorical",
                        preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        cat,  # returns the categorical feature indices
                    ),
                    (
                        "continuous",
                        impute.SimpleImputer(strategy="median"),
                        cont,
                    ),  # returns the numeric feature indices
                ]
            ),
        ),
        ("Classifier", ensemble.RandomForestClassifier(n_estimators=10)),
    ]
)

run = openml.runs.run_model_on_task(pipe, task, avoid_duplicate_runs=False)
myrun = run.publish()
print(f"Uploaded to {myrun.openml_url}")


# The above pipeline works with the helper functions that internally deal with pandas DataFrame.
# In the case, pandas is not available, or a NumPy based data processing is the requirement, the
# above pipeline is presented below to work with NumPy.

# Extracting the indices of the categorical columns
features = task.get_dataset().features
categorical_feature_indices = []
numeric_feature_indices = []
for i in range(len(features)):
    if features[i].name == task.target_name:
        continue
    if features[i].data_type == "nominal":
        categorical_feature_indices.append(i)
    else:
        numeric_feature_indices.append(i)

pipe = pipeline.Pipeline(
    steps=[
        (
            "Preprocessing",
            compose.ColumnTransformer(
                [
                    (
                        "categorical",
                        preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        categorical_feature_indices,
                    ),
                    (
                        "continuous",
                        impute.SimpleImputer(strategy="median"),
                        numeric_feature_indices,
                    ),
                ]
            ),
        ),
        ("Classifier", ensemble.RandomForestClassifier(n_estimators=10)),
    ]
)

run = openml.runs.run_model_on_task(pipe, task, avoid_duplicate_runs=False)
myrun = run.publish()
print(f"Uploaded to {myrun.openml_url}")

###############################################################################
# Running flows on tasks offline for later upload
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For those scenarios where there is no access to internet, it is possible to run
# a model on a task without uploading results or flows to the server immediately.

# To perform the following line offline, it is required to have been called before
# such that the task is cached on the local openml cache directory:
task = openml.tasks.get_task(96)

# The following lines can then be executed offline:
run = openml.runs.run_model_on_task(
    pipe,
    task,
    avoid_duplicate_runs=False,
    upload_flow=False,
)

# The run may be stored offline, and the flow will be stored along with it:
run.to_filesystem(directory="myrun")

# They may be loaded and uploaded at a later time
run = openml.runs.OpenMLRun.from_filesystem(directory="myrun")
run.publish()

# Publishing the run will automatically upload the related flow if
# it does not yet exist on the server.

############################################################################
# Alternatively, one can also directly run flows.

# Get a task
task = openml.tasks.get_task(403)

# Build any classifier or pipeline
clf = tree.ExtraTreeClassifier()

# Obtain the scikit-learn extension interface to convert the classifier
# into a flow object.
extension = openml.extensions.get_extension_by_model(clf)
flow = extension.model_to_flow(clf)

run = openml.runs.run_flow_on_task(flow, task)

############################################################################
# Challenge
# ^^^^^^^^^
#
# Try to build the best possible models on several OpenML tasks,
# compare your results with the rest of the class and learn from
# them. Some tasks you could try (or browse openml.org):
#
# * EEG eye state: data_id:`1471 <https://www.openml.org/d/1471>`_,
#   task_id:`14951 <https://www.openml.org/t/14951>`_
# * Volcanoes on Venus: data_id:`1527 <https://www.openml.org/d/1527>`_,
#   task_id:`10103 <https://www.openml.org/t/10103>`_
# * Walking activity: data_id:`1509 <https://www.openml.org/d/1509>`_,
#   task_id:`9945 <https://www.openml.org/t/9945>`_, 150k instances.
# * Covertype (Satellite): data_id:`150 <https://www.openml.org/d/150>`_,
#   task_id:`218 <https://www.openml.org/t/218>`_, 500k instances.
# * Higgs (Physics): data_id:`23512 <https://www.openml.org/d/23512>`_,
#   task_id:`52950 <https://www.openml.org/t/52950>`_, 100k instances, missing values.

# Easy benchmarking:
for task_id in [115]:  # Add further tasks. Disclaimer: they might take some time
    task = openml.tasks.get_task(task_id)
    data = openml.datasets.get_dataset(task.dataset_id)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)

    run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)
    myrun = run.publish()
    print(f"kNN on {data.name}: {myrun.openml_url}")


############################################################################
openml.config.stop_using_configuration_for_example()
