# %% [markdown]
# A brief example on how to use tasks from OpenML.

# %%

import openml

# %% [markdown]
# Get a [task](https://docs.openml.org/concepts/tasks/) for
# [supervised classification on credit-g](https://www.openml.org/search?type=task&id=31&source_data.data_id=31):

# %%
task = openml.get("task", 31)

# Legacy path still works:
# task = openml.tasks.get_task(31)

# %% [markdown]
# Get the dataset and its data from the task.

# %%
dataset = task.get_dataset()
X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name)

# %% [markdown]
# Get the first out of the 10 cross-validation splits from the task.

# %%
train_indices, test_indices = task.get_train_test_split_indices(fold=0)
print(train_indices[:10])  # print the first 10 indices of the training set
