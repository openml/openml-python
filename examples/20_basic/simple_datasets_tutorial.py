"""
========
Datasets
========

A basic tutorial on how to list, load and visualize datasets.
"""
############################################################################
# In general, we recommend working with tasks, so that the results can
# be easily reproduced. Furthermore, the results can be compared to existing results
# at OpenML. However, for the purposes of this tutorial, we are going to work with
# the datasets directly.

# License: BSD 3-Clause

import openml

############################################################################
# List datasets
# =============

datasets_df = openml.datasets.list_datasets(output_format="dataframe")
print(datasets_df.head(n=10))

############################################################################
# Download a dataset
# ==================

# Iris dataset https://www.openml.org/d/61
dataset = openml.datasets.get_dataset(61)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:500])

############################################################################
# Load a dataset
# ==============

# X - An array/dataframe where each row represents one example with
# the corresponding feature values.
# y - the classes for each example
# categorical_indicator - an array that indicates which feature is categorical
# attribute_names - the names of the features for the examples (X) and
# target feature (y)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

############################################################################
# Tip: you can get a progress bar for dataset downloads, simply set it in
# the configuration. Either in code or in the configuration file
# (see also the introduction tutorial)

openml.config.show_progress = True


############################################################################
# Visualize the dataset
# =====================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


# We combine all the data so that we can map the different
# examples to different colors according to the classes.
combined_data = pd.concat([X, y], axis=1)
iris_plot = sns.pairplot(combined_data, hue="class")
iris_plot.map_upper(hide_current_axis)
plt.show()
