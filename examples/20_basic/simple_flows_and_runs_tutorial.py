# %% [markdown]
# # Flows and Runs
# A simple tutorial on how to train/run a model and how to upload the results.

# %%
import openml
from sklearn import ensemble, neighbors

from openml.utils import thread_safe_if_oslo_installed


# %% [markdown]
# <div class="admonition warning">
#     <p class="admonition-title">Warning</p>
#     <p>
#         This example uploads data. For that reason, this example connects to the
#         test server at <a href="https://test.openml.org"
#         target="_blank">test.openml.org</a>.<br>
#         This prevents the main server from becoming overloaded with example datasets, tasks,
#         runs, and other submissions.<br>
#         Using this test server may affect the behavior and performance of the
#         OpenML-Python API.
#     </p>
# </div>

# %%
openml.config.start_using_configuration_for_example()

# %% [markdown]
# ## Train a machine learning model

# NOTE: We are using dataset 20 from the test server: https://test.openml.org/d/20

# %%
dataset = openml.datasets.get_dataset(20)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)
if y is None:
    y = X["class"]
    X = X.drop(columns=["class"], axis=1)
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
