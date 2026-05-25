# %% [markdown]
# A basic tutorial on how to list, load and visualize datasets.
#
# In general, we recommend working with tasks, so that the results can
# be easily reproduced. Furthermore, the results can be compared to existing results
# at OpenML. However, for the purposes of this tutorial, we are going to work with
# the datasets directly.

# %%

import openml

# %% [markdown]
# ## List datasets stored on OpenML

# %%
datasets_df = openml.list_all("dataset")
print(datasets_df.head(n=10))

# Legacy path still works:
# datasets_df = openml.datasets.list_datasets()

# %% [markdown]
# ## Download a dataset

# %%
# Iris dataset https://www.openml.org/d/61
dataset = openml.get(61)
# You can also fetch by name:
# dataset = openml.get("Fashion-MNIST")

# Legacy path still works:
# dataset = openml.datasets.get_dataset(dataset_id=61)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is '{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:500])

# %% [markdown]
# ## Load a dataset
# * `X` - A dataframe where each row represents one example with
#   the corresponding feature values.
# * `y` - the classes for each example
# * `categorical_indicator` - a list that indicates which feature is categorical
# * `attribute_names` - the names of the features for the examples (X) and
# target feature (y)

# %%
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)

# %% [markdown]
# Visualize the dataset

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris_plot = sns.pairplot(pd.concat([X, y], axis=1), hue="class")
plt.show()
