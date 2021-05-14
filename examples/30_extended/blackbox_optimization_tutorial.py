"""
Blackbox Optimization
=====================
"""

# License: BSD 3-Clause

import openml
from openml.runs.functions import format_prediction
from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration

import time
import numpy as np
from collections import OrderedDict
from bayes_opt import BayesianOptimization
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = openml.evaluations.list_evaluations_setups(
    function="predictive_accuracy",
    flows=[8353],
    tasks=[6],
    output_format="dataframe",
    # Using this flag incorporates the hyperparameters into the returned dataframe. Otherwise,
    # the dataframe would contain a field ``paramaters`` containing an unparsed dictionary.
    parameters_in_separate_columns=True,
)
hyperparameters = ["sklearn.svm.classes.SVC(16)_C", "sklearn.svm.classes.SVC(16)_gamma"]
df = df[hyperparameters]
C = df["sklearn.svm.classes.SVC(16)_C"].astype(float)
C_bound = (C.min(), C.max())
gamma = df["sklearn.svm.classes.SVC(16)_gamma"].astype(float)
gamma_bound = (gamma.min(), gamma.max())

# Configuration search space or bounds
bounds = [C_bound, gamma_bound]
pbounds = dict(C=C_bound, gamma=gamma_bound)

############################################################################
# We'll use the test server for the rest of this tutorial.
#
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt
openml.config.start_using_configuration_for_example()

############################################################################
# Determining seed to be used across data splits and DE optimisation
seed = 123

# Preparing data splits
# task_id = 1200
# task = openml.tasks.get_task(task_id)  # IRIS with 33% heldout test
# print(task)

task_id = 18
task = openml.tasks.get_task(task_id)  # Letter with 33% heldout test
print(task)

# Retrieving training and test splits
training_idx, test_idx = task.get_train_test_split_indices(fold=0, repeat=0)
# Fixed train-test data splits for each OpenML tasks
print(len(training_idx), len(test_idx))

# Fetching and creating Training-Validation-Test data splits
X, y = task.get_X_and_y(dataset_format="dataframe")
train_X = X.iloc[training_idx]
train_y = y.iloc[training_idx]
test_X = X.iloc[test_idx]
test_y = y.iloc[test_idx]

train_X, valid_X, train_y, valid_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=seed
)

############################################################################
# Function that will be called to optimize by DE


def svm_wrapper(C, gamma, train_X, train_y, valid_X, valid_y):
    clf = SVC(C=C, gamma=gamma)
    clf.fit(train_X, train_y)
    # BO maximizes accuracy
    return accuracy_score(clf.predict(valid_X), valid_y)


# Initializing BO optimizer
optimizer = BayesianOptimization(
    f=lambda C, gamma: svm_wrapper(C, gamma, train_X, train_y, valid_X, valid_y),
    pbounds=pbounds,
    random_state=seed,
)
# Running BO optimization for 10 iterations after seeding model with 3 random evaluations
# A total of 13 function evaluations
optimizer.maximize(init_points=3, n_iter=10)

# Best configuration
best_config = optimizer.max["params"]
print("Best found configuration: \n{}".format(best_config))
print("Best accuracy obtained on validation set : {:.5f} %".format(optimizer.max["target"] * 100))

model_params = dict(**best_config, random_state=seed, probability=True)

# Obtaining opimization trace and building an OpenMLRunTrace object for the run
# Following these guidelines: https://docs.openml.org/OpenML_definition/#trace
trace_iterations = dict()
for i, _eval in enumerate(optimizer.res):
    selected = False
    if _eval == optimizer.max:
        selected = True
    trace_iterations[(0, 0, i)] = OpenMLTraceIteration(
        repeat=0,
        fold=0,
        iteration=i,
        setup_string="",
        evaluation=_eval["target"],
        selected=selected,
        parameters=OrderedDict(_eval["params"]),
    )
run_trace = OpenMLRunTrace(-1, trace_iterations=trace_iterations)

###############################################################################
# Re-fitting model on complete training data and obtaining predictions locally

clf = SVC(**model_params)
clf.fit(train_X.append(valid_X), train_y.append(valid_y))
_predictions = clf.predict(test_X)
print("Accuracy on the test set: {:.5f} %".format(accuracy_score(_predictions, test_y) * 100))

############################################################################
# Preparing flow

# svm_on_test_server = [
#     101323, 101343, 101383, 103511, 103551, 103591, 103632, 204904, 207060, 219404, 220130
# ]
# flow_id = svm_on_test_server[np.random.randint(len(svm_on_test_server))]
# flow = openml.flows.get_flow(flow_id)
# print(flow)

general = dict(
    name="sklearn_bbo_example_flow",
    description=("Running BO on SVM using OpenML"),
    external_version="bayesian-optimization==1.2.0",
    language="English",
    tags=["bbo", "svc", "sklearn"],
    dependencies="bayesian-optimization==1.2.0",
    components=OrderedDict()
)

str_model_params = {}
for k, v in model_params.items():
    str_model_params[k] = str(v)

flow_hyperparameters = dict(
    parameters=OrderedDict(**str_model_params),
    parameters_meta_info=OrderedDict(
        C=OrderedDict(
            description="Regularization parameter. "
                        "The strength of the regularization is inversely proportional to C.",
            data_type="float"
        ),
        gamma=OrderedDict(
            description="Kernel coefficient for rbf, poly and sigmoid.",
            data_type="float"
        ),
        random_state=OrderedDict(
            description="Controls the pseudo random number generation "
                        "for shuffling the data for probability estimates",
            data_type="integer"
        ),
        probability=OrderedDict(description="To enable probability estimates", data_type="bool"),
    )
)
flow = openml.flows.OpenMLFlow(**general, **flow_hyperparameters, model=clf)
flow.publish()
flow_id = flow.flow_id

############################################################################
# Using OpenML-Python API to create runs and obtain predictions

# clf = SVC(**model_params)
# # Updating flow parameters with best found configuration from DE
# flow.parameters.update(model_params)
# run = openml.runs.run_flow_on_task(
#     flow=flow, task=task, upload_flow=False, seed=seed, avoid_duplicate_runs=False
# )
# run.publish()
#
# # Fetching model specifications (best configuration) from uploaded run
# _clf = openml.runs.initialize_model_from_run(run.run_id)
# _clf.probability = True
# _clf.random_state = seed
# _clf.fit(train_X.append(valid_X), train_y.append(valid_y))
# predictions_2 = _clf.predict(test_X)
#
# # Check if the predictions obtained locally and from the uploaded run are the same
# assert np.all(predictions_1 == predictions_2)

############################################################################
# Creating custom run object

# obtaining predictions and predicition probabilities on the OpenML test set
y_pred = _predictions.copy()
y_proba = clf.predict_proba(test_X)

# defining flow parameters used for the model
parameters = [
    OrderedDict([
        ("oml:name", "C"), ("oml:value", model_params["C"]), ("oml:component", flow_id)
    ]),
    OrderedDict([
        ("oml:name", "gamma"), ("oml:value", model_params["gamma"]), ("oml:component", flow_id)
    ])
]

# collating predictions for the OpenML arff data to be stored on the server
predictions = []
for i in range(len(test_idx)):
    prediction = format_prediction(
        task=task,
        repeat=0,
        fold=0,
        index=test_idx[i],
        prediction=y_pred[i],
        truth=test_y[test_idx[i]],
        proba={c: pb for (c, pb) in zip(task.class_labels, y_proba[i])}
    )
    predictions.append(prediction)

# Creating OpenML run object
benchmark_command = "Run DE on SVM for IRIS"
my_run = openml.runs.OpenMLRun(
    task_id=task_id,
    flow_id=flow_id,
    dataset_id=task.dataset_id,
    parameter_settings=parameters,
    setup_string=benchmark_command,
    data_content=predictions,
    trace=run_trace,  # appending BO optimization trace to the run
    tags=["bbo", "de", "svc", "sklearn"],
    description_text="Run generated by SVM tuned by DE.",
)
my_run.publish()
print(my_run)

############################################################################
# Comparing and checking if runs from both methods yield same result

# print("[Case 1 - OpenML API run] Accuracy on the test set: {:.5f}".format(
#     accuracy_score(predictions_1, test_y))
# )

# wait for OpenML servers to compute metrics
evaluations = None
while evaluations is None or evaluations == OrderedDict():
    evaluations = openml.evaluations.list_evaluations(
        function='predictive_accuracy', runs=[my_run.run_id]
    )
    time.sleep(1)
print("Accuracy on the test set computed by OpenML: {:.5f} %".format(
    evaluations[my_run.run_id].value * 100)
)


openml.config.stop_using_configuration_for_example()
