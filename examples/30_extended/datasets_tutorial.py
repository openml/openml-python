"""
========
Datasets
========

How to list and download datasets.
"""

# License: BSD 3-Clauses

import openml
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset

############################################################################
# Exercise 0
# **********
#
# * List datasets
#
#   * Use the output_format parameter to select output type
#   * Default gives 'dict' (other option: 'dataframe', see below)
#
# Note: list_datasets will return a pandas dataframe by default from 0.15. When using
# openml-python 0.14, `list_datasets` will warn you to use output_format='dataframe'.
openml_list = openml.datasets.list_datasets()  # returns a dict

# Show a nice table with some key data properties
datalist = pd.DataFrame.from_dict(openml_list, orient="index")
datalist = datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

print(f"First 10 of {len(datalist)} datasets...")
datalist.head(n=10)

# The same can be done with lesser lines of code
openml_df = openml.datasets.list_datasets(output_format="dataframe")
openml_df.head(n=10)

############################################################################
# Exercise 1
# **********
#
# * Find datasets with more than 10000 examples.
# * Find a dataset called 'eeg_eye_state'.
# * Find all datasets with more than 50 classes.
datalist[datalist.NumberOfInstances > 10000].sort_values(["NumberOfInstances"]).head(n=20)
""
datalist.query('name == "eeg-eye-state"')
""
datalist.query("NumberOfClasses > 50")

############################################################################
# Download datasets
# =================

# This is done based on the dataset ID.
dataset = openml.datasets.get_dataset(1471)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:500])

############################################################################
# Get the actual data.
#
# The dataset can be returned in 3 possible formats: as a NumPy array, a SciPy
# sparse matrix, or as a Pandas DataFrame. The format is
# controlled with the parameter ``dataset_format`` which can be either 'array'
# (default) or 'dataframe'. Let's first build our dataset from a NumPy array
# and manually create a dataframe.
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
eeg = pd.DataFrame(X, columns=attribute_names)
eeg["class"] = y
print(eeg[:10])

############################################################################
# Instead of manually creating the dataframe, you can already request a
# dataframe with the correct dtypes.
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)
print(X.head())
print(X.info())

############################################################################
# Sometimes you only need access to a dataset's metadata.
# In those cases, you can download the dataset without downloading the
# data file. The dataset object can be used as normal.
# Whenever you use any functionality that requires the data,
# such as `get_data`, the data will be downloaded.
dataset = openml.datasets.get_dataset(1471, download_data=False)

############################################################################
# Exercise 2
# **********
# * Explore the data visually.
eegs = eeg.sample(n=1000)
_ = pd.plotting.scatter_matrix(
    eegs.iloc[:100, :4],
    c=eegs[:100]["class"],
    figsize=(10, 10),
    marker="o",
    hist_kwds={"bins": 20},
    alpha=0.8,
    cmap="plasma",
)


############################################################################
# Edit a created dataset
# ======================
# This example uses the test server, to avoid editing a dataset on the main server.
#
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt
openml.config.start_using_configuration_for_example()
############################################################################
# Edit non-critical fields, allowed for all authorized users:
# description, creator, contributor, collection_date, language, citation,
# original_data_url, paper_url
desc = (
    "This data sets consists of 3 different types of irises' "
    "(Setosa, Versicolour, and Virginica) petal and sepal length,"
    " stored in a 150x4 numpy.ndarray"
)
did = 128
data_id = edit_dataset(
    did,
    description=desc,
    creator="R.A.Fisher",
    collection_date="1937",
    citation="The use of multiple measurements in taxonomic problems",
    language="English",
)
edited_dataset = get_dataset(data_id)
print(f"Edited dataset ID: {data_id}")


############################################################################
# Editing critical fields (default_target_attribute, row_id_attribute, ignore_attribute) is allowed
# only for the dataset owner. Further, critical fields cannot be edited if the dataset has any
# tasks associated with it. To edit critical fields of a dataset (without tasks) owned by you,
# configure the API key:
# openml.config.apikey = 'FILL_IN_OPENML_API_KEY'
# This example here only shows a failure when trying to work on a dataset not owned by you:
try:
    data_id = edit_dataset(1, default_target_attribute="shape")
except openml.exceptions.OpenMLServerException as e:
    print(e)

############################################################################
# Fork dataset
# ============
# Used to create a copy of the dataset with you as the owner.
# Use this API only if you are unable to edit the critical fields (default_target_attribute,
# ignore_attribute, row_id_attribute) of a dataset through the edit_dataset API.
# After the dataset is forked, you can edit the new version of the dataset using edit_dataset.

data_id = fork_dataset(1)
print(data_id)
data_id = edit_dataset(data_id, default_target_attribute="shape")
print(f"Forked dataset ID: {data_id}")

openml.config.stop_using_configuration_for_example()
