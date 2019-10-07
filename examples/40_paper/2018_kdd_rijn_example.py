"""
van Rijn and Hutter (2018)
==========================

A tutorial on how to reproduce the paper *Hyperparameter Importance Across Datasets*.

Publication
~~~~~~~~~~~

| Hyperparameter importance across datasets
| Jan N. van Rijn and Frank Hutter
| In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2018
| Available at https://dl.acm.org/citation.cfm?id=3220058
"""
import fanova
import json
import logging
import matplotlib.pyplot as plt
import openml
import pandas as pd
import seaborn as sns

root = logging.getLogger()
root.setLevel(logging.INFO)

##############################################################################
# With the advent of automated machine learning, automated hyperparameter
# optimization methods are by now routinely used in data mining. However, this
# progress is not yet matched by equal progress on automatic analyses that
# yield information beyond performance-optimizing hyperparameter settings.
# In this example, we aim to answer the following two questions: Given an
# algorithm, what are generally its most important hyperparameters?
#
# This work is carried out on the OpenML-100 benchmark suite, which can be
# obtained by `openml.study.get_suite('OpenML100')`. In this example, we
# conduct the experiment on the Support Vector Machine (`flow_id=7707`)
# with specific kernel (we will perform a post-process filter operation for
# this). We should set some other experimental parameters (number of results
# per task, evaluation measure and the number of trees of the internal
# functional Anova) before the fun can begin.
#
# Note that we simplify the example in several ways
# 1) We only consider numerical hyperparameters
# 2) We consider all hyperparameters that are numerical (in reality, some
#    hyperparameters might be inactive (e.g., `degree`) or irrelevant (e.g.,
#    `random_state`)
# 3) We assume all hyperparameters to be on uniform scale
#
# Any difference in conclusion between the actual paper and the presented
# results is most likely due to one of these simplifications. For example,
# # the hyperparameter C looks rather insignificant, whereas it is quite
# important when it is put on a log-scale. All these simplifications can be
# addressed by defining a ConfigSpace. For a more elaborated example that uses
# this, please see:
# https://github.com/janvanrijn/openml-pimp/blob/d0a14f3eb480f2a90008889f00041bdccc7b9265/examples/plot/plot_fanova_aggregates.py # noqa F401

suite = openml.study.get_suite('OpenML100')
flow_id = 7707
parameter_filters = {
    'sklearn.svm.classes.SVC(17)_kernel': 'sigmoid'
}
evaluation_measure = 'predictive_accuracy'
limit_per_task = 500
n_trees = 16

fanova_results = []
# we will obtain all results from OpenML per task. Practice has shown that this places the bottleneck on the
# communication with OpenML, and for iterated experimenting it is better to cache the results in a local file.
for idx, task_id in enumerate(suite.tasks):
    logging.info('Starting with task %d (%d/%d)' % (task_id, idx+1, len(suite.tasks)))
    evals = openml.evaluations.list_evaluations_setups(
        evaluation_measure, flow=[flow_id], task=[task_id], size=limit_per_task, output_format='dataframe')

    performance_column = 'value'
    # make a DataFrame consisting of all hyperparameters (which is a dict in setup['parameters']) and the performance
    # value (in setup['value']). The following line looks a bit complicated, but combines 2 tasks: a) combine
    # hyperparameters and performance data in a single dict, b) cast hyperparameter values to the appropriate format
    # Note that
    try:
        setups_evals = pd.DataFrame([dict(**{name: json.loads(value) for name, value in setup['parameters'].items()},
                                          **{performance_column: setup[performance_column]})
                                     for _, setup in evals.iterrows()])
    except json.decoder.JSONDecodeError as e:
        logging.warning('Task %d error: %s' % (task_id, e))
        continue
    # apply our filters, to have only the setups that comply to the hyperparameters we want
    for filter_key, filter_value in parameter_filters.items():
        setups_evals = setups_evals[setups_evals[filter_key] == filter_value]
    # in this simplified example, we only display numerical and float hyperparameters. For categorical hyperparameters,
    # the fanova library needs to be informed by using a configspace object.
    setups_evals = setups_evals.select_dtypes(include=['int64', 'float64'])
    # drop rows with unique values. These are by definition not an interesting hyperparameter, e.g., `axis`, `verbose`
    setups_evals = setups_evals[[c for c in list(setups_evals)
                                 if len(setups_evals[c].unique()) > 1 or c == performance_column]]
    # determine x values to pass to fanova library
    parameter_names = [pname for pname in setups_evals.columns.values if pname != performance_column]
    evaluator = fanova.fanova.fANOVA(
        X=setups_evals[parameter_names].values, Y=setups_evals[performance_column].values, n_trees=n_trees)
    for idx, pname in enumerate(parameter_names):
        try:
            fanova_results.append({
                'hyperparameter': pname if len(pname) < 35 else '[...] %s' % pname[-30:],
                'fanova': evaluator.quantify_importance([idx])[(idx,)]['individual importance']
            })
        except RuntimeError as e:
            # functional ANOVA sometimes crashes with a RuntimeError, e.g., on tasks where the performance is constant
            # for all configurations (there is no variance)
            logging.warning('Task %d error: %s' % (task_id, e))

# transform fanova_results from a list of dicts into a DataFrame
fanova_results = pd.DataFrame(fanova_results)

##############################################################################
# make the boxplot of the variance contribution. Obviously, we can also use this data to make the Nemenyi plot, but this
# relies on the rather complex `Orange` dependency (`pip install Orange3`). For the complete example, the reader is
# referred to the more elaborate script (referred to earlier)
fig, ax = plt.subplots()
sns.boxplot(x='hyperparameter', y='fanova', data=fanova_results, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Variance Contribution')
ax.set_xlabel(None)
plt.tight_layout()
plt.show()
