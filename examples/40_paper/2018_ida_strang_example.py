"""
Strang et al. (2018)
====================

A tutorial on how to reproduce the analysis conducted for *Don't Rule Out Simple Models
Prematurely: A Large Scale Benchmark Comparing Linear and Non-linear Classifiers in OpenML*.

Publication
~~~~~~~~~~~

| Don't Rule Out Simple Models Prematurely: A Large Scale Benchmark Comparing Linear and Non-linear Classifiers in OpenML
| Benjamin Strang, Pieter Putten, Jan van Rijn and Frank Hutter
| In *Advances in Intelligent Data Analysis XVII 17th International Symposium*, 2018
| Available at https://link.springer.com/chapter/10.1007%2F978-3-030-01768-2_25
"""
import matplotlib.pyplot as plt
import openml
import pandas as pd

study_id = 123
# for comparing svms: flow_ids = [7754, 7756]
# for comparing nns: flow_ids = [7722, 7729]
# for comparing dts: flow_ids = [7725], differentiate on hyper-parameter value instead
classifier_family = 'SVM'
flow_ids = [7754, 7756]
measure = 'predictive_accuracy'
meta_features = ['NumberOfInstances', 'NumberOfFeatures']


evaluations = openml.evaluations.list_evaluations(measure, flow=flow_ids, study=study_id, output_format='dataframe')
evaluations['data_id'] = evaluations['data_id'].apply(pd.to_numeric)
evaluations = evaluations.pivot(index='data_id', columns='flow_id', values='value').dropna()
data_qualities = openml.datasets.list_datasets(data_id=list(evaluations.index.values), output_format='dataframe')
data_qualities = data_qualities[meta_features]
evaluations = evaluations.join(data_qualities, how='inner')

# adds column that indicates the difference between the two classifiers
evaluations['difference'] = evaluations[flow_ids[0]] - evaluations[flow_ids[1]]

fig_splot, ax_splot = plt.subplots()
ax_splot.plot(range(len(evaluations)), sorted(evaluations['difference']))
ax_splot.set_title(classifier_family)
ax_splot.set_xlabel('Dataset (sorted)')
ax_splot.set_ylabel('difference between linear and non-linear classifier')
ax_splot.grid(linestyle='--', axis='y')
plt.show()


# adds column that indicates the difference between the two classifiers
class_values = ['non-linear better', 'linear better', 'equal']
def determine_class(val_lin, val_nonlin):
    if val_lin < val_nonlin:
        return class_values[0]
    elif val_nonlin < val_lin:
        return class_values[1]
    else:
        return class_values[2]


evaluations['class'] = evaluations.apply(lambda row: determine_class(row[flow_ids[0]], row[flow_ids[1]]), axis=1)

fig_scatter, ax_scatter = plt.subplots()
for class_val in class_values:
    df_class = evaluations[evaluations['class'] == class_val]
    plt.scatter(df_class[meta_features[0]], df_class[meta_features[1]], label=class_val)
ax_scatter.set_title(classifier_family)
ax_scatter.set_xlabel(meta_features[0])
ax_scatter.set_ylabel(meta_features[1])
ax_scatter.legend()
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
plt.show()
