"""
====================================================
Hyperparameter Optimization Benchmark with OptunaHub
====================================================

In this tutorial, we walk through how to conduct hyperparameter optimization experiments using OpenML and OptunaHub.
"""
############################################################################
# Please make sure to install the dependencies with:
# ``pip install openml optunahub hebo`` and ``pip install --upgrade pymoo``
# Then we import all the necessary modules.

# License: BSD 3-Clause

import openml
from openml.extensions.sklearn import cat
from openml.extensions.sklearn import cont
import optuna
import optunahub
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

############################################################################
# Prepare for preprocessors and an OpenML task
# ============================================

# https://www.openml.org/search?type=study&study_type=task&id=218
task_id = 10101
seed = 42
categorical_preproc = ("categorical", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat)
numerical_preproc = ("numerical", SimpleImputer(strategy="median"), cont)
preproc = ColumnTransformer([categorical_preproc, numerical_preproc])

############################################################################
# Define a pipeline for the hyperparameter optimization
# =====================================================

# Since we use `OptunaHub <https://hub.optuna.org/>`__ for the benchmarking of hyperparameter optimization,
# we follow the `Optuna <https://github.com/optuna/optuna/>`__ search space design.
# We can simply pass the parametrized classifier to `run_model_on_task` to obtain the performance of the pipeline
# on the specified OpenML task.

def objective(trial: optuna.Trial) -> Pipeline:
    clf = RandomForestClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 32, log=True),
        min_samples_leaf=trial.suggest_float("min_samples_leaf", 0.0, 1.0),
        random_state=seed,
    )
    pipe = Pipeline(steps=[("preproc", preproc), ("model", clf)])
    run = openml.runs.run_model_on_task(pipe, task=task_id, avoid_duplicate_runs=False)
    accuracy = max(run.fold_evaluations["predictive_accuracy"][0].values())    
    return accuracy

############################################################################
# Load a sampler from OptunaHub
# =============================

# OptunaHub is a feature-sharing plotform for hyperparameter optimization methods.
# For example, we load a state-of-the-art algorithm (`HEBO <https://github.com/huawei-noah/HEBO/tree/master/HEBO>`__
# , the winning solution of `NeurIPS 2020 Black-Box Optimisation Challenge <https://bbochallenge.com/leaderboard/>`__)
# from OptunaHub here.

sampler = optunahub.load_module("samplers/hebo").HEBOSampler(seed=seed)

############################################################################
# Optimize the pipeline
# =====================

# We now run the optimization. For more details about Optuna API,
# please visit `the API reference <https://optuna.readthedocs.io/en/stable/reference/index.html>`__.

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=15)

############################################################################
# Visualize the optimization history
# ==================================

# It is very simple to visualize the result by the Optuna visualization module.
# For more details, please check `the API reference <https://optuna.readthedocs.io/en/stable/reference/visualization/index.html>`__.

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
