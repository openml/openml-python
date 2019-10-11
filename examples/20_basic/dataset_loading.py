"""
Dataset Loading
===============

An example on how to load a dataset and visualize the data.
"""
############################################################################
import openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

# We recommend to work with tasks so that the results can be easily
# reproducible and also compared to the previous results on OpenML.
# However, for the purposes of this tutorial we are going to work with
# the dataseet.
# Iris dataset https://www.openml.org/d/61
dataset = openml.datasets.get_dataset(61)

# X - An array/dataframe where each row represents one example with
# the corresponding feature values.
# y - the classes for each example
# categorical_indicator - an array that indicates which feature is categorical
# attribute_names - the names of the features for the examples (X) and
# target feature (y)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)

# We combine all the data so that we can map the different
# examples to different colors according to the classes.
combined_data = pd.concat([X, y], axis=1)
iris_plot  = sns.pairplot(combined_data, hue="class")
iris_plot.map_upper(hide_current_axis)
plt.show()
