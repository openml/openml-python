# %% [markdown]
# # Datasets
# A basic tutorial on how to list, load and visualize datasets.
#
# In general, we recommend working with tasks, so that the results can
# be easily reproduced. Furthermore, the results can be compared to existing results
# at OpenML. However, for the purposes of this tutorial, we are going to work with
# the datasets directly.

# %%

import openml

# %% [markdown]
# ## List datasets

# %%
datasets_df = openml.datasets.list_datasets(output_format="dataframe")
print(datasets_df.head(n=10))

# %% [markdown]
# ## Download a dataset

# %%
# Iris dataset https://www.openml.org/d/61
dataset = openml.datasets.get_dataset(dataset_id=61, version=1)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:500])

# %% [markdown]
# ## Load a dataset
# X - An array/dataframe where each row represents one example with
# the corresponding feature values.
#
# y - the classes for each example
#
# categorical_indicator - an array that indicates which feature is categorical
#
# attribute_names - the names of the features for the examples (X) and
# target feature (y)

# %%
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)

# %% [markdown]
# Visualize the dataset

<<<<<<< docs/mkdoc -- Incoming Change
# %%
=======
import matplotlib.pyplot as plt
>>>>>>> develop -- Current Change
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


# We combine all the data so that we can map the different
# examples to different colors according to the classes.
combined_data = pd.concat([X, y], axis=1)
iris_plot = sns.pairplot(combined_data, hue="class")
iris_plot.map_upper(hide_current_axis)
plt.show()

# License: BSD 3-Clause
