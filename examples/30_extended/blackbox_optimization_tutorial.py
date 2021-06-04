"""
================================
Blackbox Optimization and OpenML
================================

OpenML was designed to be an online repository standardizing and democratizing the availability
of machine learning datasets, their data splits, models with their hyperparameters, and the
associated runs and evaluations. Such a setup allows for reproducibility in not just standard
machine learning use, but also cases such as black-box optimization. In this example, we treat the
problem of tuning or optimizing a machine learning model's hyperparameters for a particular dataset
as a black-box optimization problem. We shall tune 2 hyperparameters of an SVM model, to obtain a
configuration that performs the best on a held-out validation set.

OpenML tasks have predetermined train-test splits. The onus is on the user to create a validation
set on which optimization performance can be evaluated. In this tutorial, we split the OpenML
training split further to create a validation split on which the optimization is evaluated. The
best found configuration is then refit on the OpenML training split (training + validation split)
and evaluated on the test split, which is then uploaded to the OpenML server as an OpenMLRun
object, along with the optimization trace from Bayesian Optimization (BO).

Additional requirement for this tutorial:

* bayesian-optimization >= 1.2.0

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


############################################################################
# We'll switch to the test server for this tutorial.
#
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt
openml.config.start_using_configuration_for_example()

############################################################################
# Preparing data splits
# *********************
# Firstly the dataset needs to be retrieved from OpenML. The train-test split
# that already exists will remain as is. We shall take the training set and
# using a seed (for reproducibility) split it into training and validation
# sets. The latter is going to be used for evaluation by the BO loop.

# Determining seed to be used across data splits and DE optimisation
seed = 123

task_id = 18  # Letter with a 33% heldout test set
task = openml.tasks.get_task(task_id)
print(task)

# Retrieving training and test splits
training_idx, test_idx = task.get_train_test_split_indices(fold=0, repeat=0)

# Fixed train-test data splits for each OpenML tasks
print("\n# training instances: {}".format(len(training_idx)))
print("# test instances: {}".format(len(test_idx)))

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
# Designing hyperparameter space to optimize
# ******************************************
# We shall create a 2-dimensional hyperparameter space here for the C and gamma
# parameters from this `SVM guide <https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf>`_.

C_bound = (2 ** -5, 2 ** 15)
gamma_bound = (2 ** -15, 2 ** 3)

# Configuration search space or bounds
bounds = [C_bound, gamma_bound]
pbounds = dict(C=C_bound, gamma=gamma_bound)


############################################################################
# Defining model to optimize using BO
# ***********************************
# The BO loop needs a black-box function to call, to which it can provide
# the hyperparameter configuration to be evaluated. The function returns
# only a float as the response. In this case, the BO loop maximizes and
# therefore the black-box function defined returns the accuracy.
#
# The BO loop only provides the black box function with the values for
# the 2 hyperparameters - C, gamma. The other necessary input to the model
# can either be passed as default arguments to the function definition. Or
# a lambda function can be used in the BO function call to specify the extra
# input parameters. We choose the latter here.


def svm_wrapper(C, gamma, train_X, train_y, valid_X, valid_y):
    clf = SVC(C=C, gamma=gamma)
    clf.fit(train_X, train_y)
    # BO maximizes accuracy
    return accuracy_score(clf.predict(valid_X), valid_y)


############################################################################
# Running Bayesian Optimization
# *****************************
# We use the popular BO package `bayes_opt <https://github.com/fmfn/BayesianOptimization>`_.
# We warm-start the BO loop with 3 random configurations to begin with.

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

############################################################################
# Obtaining optimization trace
# ****************************
# In general, an OpenMLRunTrace object contains a record of the sequence of
# evaluations performed at each iteration. Each such record is an object of
# type OpenMLTraceIteration which maps the fold number, the repetition count,
# to the model parameters used to obtain the score at that iteration. If the
# score at that iteration is the best seen score, the OpenMLTraceIteration also
# records if those model parameters were selected during the sequence of optimization.
# In the context of this tutorial, the OpenMLRunTrace will contain information
# about all the function evaluations performed during BO to find the best
# performing configuration.

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

############################################################################
# Refitting model with best configuration
# ***************************************
# Re-fitting model on complete training data and obtaining predictions locally

model_params = dict(**best_config, random_state=seed, probability=True)

clf = SVC(**model_params)
clf.fit(train_X.append(valid_X), train_y.append(valid_y))
_predictions = clf.predict(test_X)
print("Accuracy on the test set: {:.5f} %".format(accuracy_score(_predictions, test_y) * 100))

###########################################
# Preparing an OpenML flow with refit model
# *****************************************
# Creating a new flow that will store the SVM model with hyperparameters
# optimized for the chosen dataset.

general = dict(
    name="sklearn_bbo_example_flow",
    description=("Running BO on SVM using OpenML"),
    external_version="bayesian-optimization==1.2.0, sklearn==0.24.1",
    language="English",
    tags=["bbo", "svc", "svm", "sklearn", "bayesopt", "hpo", "bayesian-optimization"],
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

###########################################
# Creating a custom OpenML run object
# ***********************************
# We have all OpenML components (task, flow) in place to record an evaluation. In this section,
# we use the previously obtained predictions on the test-split for the optimized SVM model. This
# information can be bundled in the form of an OpenMLRun object and uploaded to OpenML.

# Obtaining predictions and predicition probabilities on the OpenML test set
y_pred = _predictions.copy()
y_proba = clf.predict_proba(test_X)

# Defining flow parameters used for the model
parameters = [
    OrderedDict([
        ("oml:name", "C"), ("oml:value", model_params["C"]), ("oml:component", flow_id)
    ]),
    OrderedDict([
        ("oml:name", "gamma"), ("oml:value", model_params["gamma"]), ("oml:component", flow_id)
    ])
]

# Collating predictions for the OpenML arff data to be stored on the server
predictions = []
for i in range(len(test_idx)):
    # openml.runs.functions.format_prediction() is a helper function in OpenML-Python
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
# On uploading a run, OpenML servers compute performance metrics on the
# uploaded predictions too. Such evaluations for the relevant runs can be
# obtained at any point, by anyone.

# Wait for OpenML servers to compute metrics
evaluations = None
while evaluations is None or evaluations == OrderedDict():
    evaluations = openml.evaluations.list_evaluations(
        function='predictive_accuracy', runs=[my_run.run_id]
    )
    time.sleep(1)
print("Accuracy on the test set computed by OpenML: {:.5f} %".format(
    evaluations[my_run.run_id].value * 100)
)

############################################################################
# A minor drawback of creating a flow from scratch is that this uploaded run
# cannot be fetched from the server to instantiate an SVM model with the
# best found parameters directly. OpenML requires that all models be serialized
# with the help of `OpenML extensions <http://openml.github.io/openml-python/main/extensions.html>`_
# for the machine learning library the model is being called from.
# However, the same can be performed using a custom flow, without an extension,
# albeit with few extra steps as shown below:

run = openml.runs.get_run(my_run.run_id)

model_params = dict(random_state=seed, probability=True)
for _, elem in enumerate(run.parameter_settings):
    model_params[elem['oml:name']] = float(elem['oml:value'])

# Initializing model with the parameters obtained from the run object
clf = SVC(**model_params)
print(clf)


openml.config.stop_using_configuration_for_example()
