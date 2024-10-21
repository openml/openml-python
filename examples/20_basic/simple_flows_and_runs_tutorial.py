# %% [markdown]
# # Flows and Runs
# A simple tutorial on how to train/run a model and how to upload the results.

# %%
import openml
from sklearn import ensemble, neighbors


# %% [markdown]
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt

# %%
openml.config.start_using_configuration_for_example()

# %% [markdown]
# ## Train a machine learning model

# NOTE: We are using dataset 20 from the test server: https://test.openml.org/d/20

# %%
dataset = openml.datasets.get_dataset(20)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)
clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)

# %% [markdown]
# ## Running a model on a task

# %%
task = openml.tasks.get_task(119)
clf = ensemble.RandomForestClassifier()
run = openml.runs.run_model_on_task(clf, task)
print(run)

# %% [markdown]
# ## Publishing the run

# %%
myrun = run.publish()
print(f"Run was uploaded to {myrun.openml_url}")
print(f"The flow can be found at {myrun.flow.openml_url}")

# %%
openml.config.stop_using_configuration_for_example()
# License: BSD 3-Clause
