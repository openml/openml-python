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

suite_id = 14
flow_id = 6969
function = 'predictive_accuracy'
limit_per_task = 200
n_trees = 16
root = logging.getLogger()
root.setLevel(logging.INFO)

suite = openml.study.get_suite(suite_id)

fanova_results = []
for idx, task_id in enumerate(suite.tasks):
    logging.info('Starting with task %d (%d/%d)' % (task_id, idx+1, len(suite.tasks)))
    try:
        evals = openml.evaluations.list_evaluations_setups(
            function, flow=[flow_id], task=[task_id], size=limit_per_task, output_format='dataframe')
    except json.decoder.JSONDecodeError:
        continue
    # drop non-numeric rows
    setups_evals = pd.DataFrame([dict(**setup['parameters'], **{'value': setup['value']})
                                 for _, setup in evals.iterrows()]).select_dtypes(include=['int64', 'float64'])
    # drop rows with unique vals
    setups_evals = setups_evals[[c for c in list(setups_evals) if len(setups_evals[c].unique()) > 1 or c == 'value']]
    parameter_names = [pname for pname in setups_evals.columns.values if pname != 'value']
    evaluator = fanova.fanova.fANOVA(X=setups_evals[parameter_names].values,
                                     Y=setups_evals['value'].values,
                                     n_trees=n_trees)
    for idx, pname in enumerate(parameter_names):
        try:
            fanova_results.append({
                'hyperparameter': pname if len(pname) < 35 else '[...] %s' % pname[-30:],
                'fanova': evaluator.quantify_importance([idx])[(idx,)]['individual importance']
            })
        except RuntimeError:
            continue

fanova_results = pd.DataFrame(fanova_results)


fig, ax = plt.subplots()
sns.boxplot(x='hyperparameter', y='fanova', data=fanova_results, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Variance Contribution')
ax.set_xlabel(None)
plt.tight_layout()
plt.show()
