# %% [markdown]
# # Introduction tutorial & Setup
# An example how to set up OpenML-Python followed up by a simple example.

# %% [markdown]
# OpenML is an online collaboration platform for machine learning which allows
# you to: Pueter
#
# * Find or share interesting, well-documented datasets
# * Define research / modelling goals (tasks)
# * Explore large amounts of machine learning algorithms, with APIs in Java, R, Python
# * Log and share reproducible experiments, models, results
# * Works seamlessly with scikit-learn and other libraries
# * Large scale benchmarking, compare to state of the art
#

# %% [markdown]
# # Installation
# Installation is done via ``pip``:
#
# .. code:: bash
#
#     pip install openml
#
# For further information, please check out the installation guide at
# :ref:`installation`.
#

# %% [markdown]
# Authentication
# ^^^^^^^^^^^^^^
#
# The OpenML server can only be accessed by users who have signed up on the
# OpenML platform. If you don’t have an account yet, sign up now.
# You will receive an API key, which will authenticate you to the server
# and allow you to download and upload datasets, tasks, runs and flows.
#
# * Create an OpenML account (free) on https://www.openml.org.
# * After logging in, open your account page (avatar on the top right)
# * Open 'Account Settings', then 'API authentication' to find your API key.
#
# There are two ways to permanently authenticate:
#
# * Use the ``openml`` CLI tool with ``openml configure apikey MYKEY``,
#   replacing **MYKEY** with your API key.
# * Create a plain text file **~/.openml/config** with the line
#   **'apikey=MYKEY'**, replacing **MYKEY** with your API key. The config
#   file must be in the directory ~/.openml/config and exist prior to
#   importing the openml module.
#
# Alternatively, by running the code below and replacing 'YOURKEY' with your API key,
# you authenticate for the duration of the python process.


# %%
# License: BSD 3-Clause

import openml
from sklearn import neighbors

# %% [markdown]
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt

# %%
openml.config.start_using_configuration_for_example()

# %% [markdown]
# When using the main server instead, make sure your apikey is configured.
# This can be done with the following line of code (uncomment it!).
# Never share your apikey with others.

# %%
# openml.config.apikey = 'YOURKEY'

# %% [markdown]
# # Caching
# When downloading datasets, tasks, runs and flows, they will be cached to
# retrieve them without calling the server later. As with the API key,
# the cache directory can be either specified through the config file or
# through the API:
#
# * Add the  line **cachedir = 'MYDIR'** to the config file, replacing
#   'MYDIR' with the path to the cache directory. By default, OpenML
#   will use **~/.openml/cache** as the cache directory.
# * Run the code below, replacing 'YOURDIR' with the path to the cache directory.

# %%
# Uncomment and set your OpenML cache directory
# import os
# openml.config.cache_directory = os.path.expanduser('YOURDIR')

# %% [markdown]
# # Simple Example
# Download the OpenML task for the eeg-eye-state.

# %%
task = openml.tasks.get_task(403)
data = openml.datasets.get_dataset(task.dataset_id)
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=False)
# Publish the experiment on OpenML (optional, requires an API key).
# For this tutorial, our configuration publishes to the test server
# as to not crowd the main server with runs created by examples.
myrun = run.publish()
print(f"kNN on {data.name}: {myrun.openml_url}")

# %%
openml.config.stop_using_configuration_for_example()
