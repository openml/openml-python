# %% [markdown]
# # Introduction Tutorial & Setup
# An example how to set up OpenML-Python followed up by a simple example.

# %% [markdown]
# # Installation
# Installation is done via ``pip``:
#
# ```bash
# pip install openml
# ```

# %% [markdown]
# # Authentication
#
# For certain functionality, such as uploading tasks or datasets, users have to
# sing up. Only accessing the data on OpenML does not require an account!
#
# If you don’t have an account yet, sign up now.
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
# you authenticate for the duration of the Python process.

# %%
import openml

openml.config.apikey = "YOURKEY"

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
import openml

openml.config.set_root_cache_directory("YOURDIR")