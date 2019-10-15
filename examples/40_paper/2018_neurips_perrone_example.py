"""
Perrone et al. (2018)
=====================

A tutorial on how to build a surrogate model based on OpenML data as done for *Scalable
Hyperparameter Transfer Learning* by Perrone et al..

Publication
~~~~~~~~~~~

| Scalable Hyperparameter Transfer Learning
| Valerio Perrone and Rodolphe Jenatton and Matthias Seeger and Cedric Archambeau
| In *Advances in Neural Information Processing Systems 31*, 2018
| Available at http://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning.pdf

This example demonstrates how OpenML runs can be used to construct a surrogate model.

In the following section, we shall do the following:

* Retrieve tasks and flows as used in the experiments by Perrone et al.
* Build a tabular data by fetching the evaluations uploaded to OpenML
* Impute missing values and handle categorical data before building a Random Forest model that
  maps hyperparameter values to the area under curve score
"""

############################################################################
import openml
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

user_id = 2702
############################################################################

"""
The subsequent functions are defined to fetch tasks, flows, evaluations and preprocess them into
a tabular format that can be used to build models.
"""

def fetch_evaluations(run_full=False, flow_type='svm', metric = 'area_under_roc_curve'):
    '''
    Fetch a list of evaluations based on the flows and tasks used in the experiments.

    Parameters
    ----------
    run_full : boolean
        If True, use the full list of tasks used in the paper
        If False, use 5 tasks with the smallest number of evaluations available
    flow_type : str, {'svm', 'xgboost'}
        To select whether svm or xgboost experiments are to be run
    metric : str
        The evaluation measure that is passed to openml.evaluations.list_evaluations

    Returns
    -------
    eval_df : dataframe
    task_ids : list
    flow_id : int
    '''
    # Collecting task IDs as used by the experiments from the paper
    if flow_type == 'svm' and run_full:
        task_ids = [10101, 145878, 146064, 14951, 34537, 3485, 3492, 3493, 3494, 37, 3889, 3891,
                    3899, 3902, 3903, 3913, 3918, 3950, 9889, 9914, 9946, 9952, 9967, 9971, 9976,
                    9978, 9980, 9983]
    elif flow_type == 'svm' and not run_full:
        task_ids = [9983, 3485, 3902, 3903, 145878]
    elif flow_type == 'xgboost' and run_full:
        task_ids = [10093, 10101, 125923, 145847, 145857, 145862, 145872, 145878, 145953, 145972,
                    145976, 145979, 146064, 14951, 31, 3485, 3492, 3493, 37, 3896, 3903, 3913,
                    3917, 3918, 3, 49, 9914, 9946, 9952, 9967]
    else:  #flow_type == 'xgboost' and not run_full:
        task_ids = [3903, 37, 3485, 49, 3913]

    # Fetching the relevant flow
    flow_id = 5891 if flow_type == 'svm' else 6767

    # Fetching evaluations
    eval_df = openml.evaluations.list_evaluations(function=metric, task=task_ids, flow=[flow_id],
                                                  uploader=[2702], output_format='dataframe')
    return eval_df, task_ids, flow_id


def create_table_from_evaluations(eval_df, flow_type='svm', run_count=np.iinfo(np.int64).max,
                                  metric = 'area_under_roc_curve', task_ids=None):
    '''
    Create a tabular data with its ground truth from a dataframe of evaluations.
    Optionally, can filter out records based on task ids.

    Parameters
    ----------
    eval_df : dataframe
        Containing list of runs as obtained from list_evaluations()
    flow_type : str, {'svm', 'xgboost'}
        To select whether svm or xgboost experiments are to be run
    run_count : int
        Maximum size of the table created, or number of runs included in the table
    metric : str
        The evaluation measure that is passed to openml.evaluations.list_evaluations
    task_ids : list, (optional)
        List of integers specifying the tasks to be retained from the evaluations dataframe

    Returns
    -------
    eval_table : dataframe
    values : list
    '''
    if task_ids is not None:
        eval_df = eval_df.loc[eval_df.task_id.isin(task_ids)]
    ncols = 4 if flow_type == 'svm' else 10  # ncols determine the number of hyperparameters
    if flow_type == 'svm':
        ncols = 4
        colnames = ['cost', 'degree', 'gamma', 'kernel']
    else:
        ncols = 10
        colnames = ['alpha', 'booster', 'colsample_bylevel', 'colsample_bytree', 'eta', 'lambda',
                    'max_depth', 'min_child_weight', 'nrounds', 'subsample']
    eval_df = eval_df.sample(frac=1)  # shuffling rows
    run_ids = eval_df.run_id[:run_count]
    eval_table = pd.DataFrame(np.nan, index=run_ids, columns=colnames)
    values = []
    for run_id in run_ids:
        r = openml.runs.get_run(run_id)
        params = r.parameter_settings
        for p in params:
            name, value = p['oml:name'], p['oml:value']
            if name in colnames:
                eval_table.loc[run_id, name] = value
        values.append(r.evaluations[metric])
    return eval_table, values


def impute_missing_values(eval_table, flow_type='svm'):
    # Replacing NaNs with fixed values outside the range of the parameters
    # given in the supplement material of the paper
    if flow_type == 'svm':
        eval_table.kernel.fillna("None", inplace=True)
        eval_table.fillna(-1, inplace=True)
    else:
        eval_table.booster.fillna("None", inplace=True)
        eval_table.fillna(-1, inplace=True)
    return eval_table


def preprocess(eval_table, flow_type='svm'):
    eval_table = impute_missing_values(eval_table, flow_type)
    # Encode categorical variables as one-hot vectors
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(eval_table.kernel.to_numpy().reshape(-1, 1))
    one_hots = enc.transform(eval_table.kernel.to_numpy().reshape(-1, 1)).toarray()
    if flow_type == 'svm':
        eval_table = np.hstack((eval_table.drop('kernel', 1), one_hots)).astype(float)
    else:
        eval_table = np.hstack((eval_table.drop('booster', 1), one_hots)).astype(float)
    return eval_table


#############################################################################
# Fetching the tasks and evaluations
# ==================================
# To read all the tasks and evaluations for them and collate into a table. Here, we are reading
# all the tasks and evaluations for the SVM flow and preprocessing all retrieved evaluations.

eval_df, task_ids, flow_id = fetch_evaluations(run_full=False)
X, y = create_table_from_evaluations(eval_df, run_count=1000)
X = preprocess(X)


#############################################################################
# Building a surrogate model on a task's evaluation
# =================================================
# The same set of functions can be used for a single task to retrieve a singular table which can
# be used for the surrogate model construction. We shall use the SVM flow here to keep execution
# time simple and quick.

# Selecting a task
task_id = task_ids[-1]
X, y = create_table_from_evaluations(eval_df, run_count=1000, task_ids=[task_id], flow_type='svm')
X = preprocess(X, flow_type='svm')

# Surrogate model
clf = RandomForestRegressor(n_estimators=50, max_depth=3)
clf.fit(X, y)
