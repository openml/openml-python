# %% [markdown]
# # Dataset upload tutorial
# A tutorial on how to create and upload a dataset to OpenML.

# %%
import numpy as np
import pandas as pd
import sklearn.datasets
from scipy.sparse import coo_matrix

import openml
from openml.datasets.functions import create_dataset

# %% [markdown]
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt

# %%
openml.config.start_using_configuration_for_example()

# %% [markdown]
# Below we will cover the following cases of the dataset object:
#
# * A numpy array
# * A list
# * A pandas dataframe
# * A sparse matrix
# * A pandas sparse dataframe

# %% [markdown]
# Dataset is a numpy array
# ========================
# A numpy array can contain lists in the case of dense data or it can contain
# OrderedDicts in the case of sparse data.
#
# # Prepare dataset
# Load an example dataset from scikit-learn which we will upload to OpenML.org
# via the API.

# %%
diabetes = sklearn.datasets.load_diabetes()
name = "Diabetes(scikit-learn)"
X = diabetes.data
y = diabetes.target
attribute_names = diabetes.feature_names
description = diabetes.DESCR

# %% [markdown]
# OpenML does not distinguish between the attributes and targets on the data
# level and stores all data in a single matrix.
#
# The target feature is indicated as meta-data of the dataset (and tasks on
# that data).

# %%
data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
attribute_names = list(attribute_names)
attributes = [(attribute_name, "REAL") for attribute_name in attribute_names] + [
    ("class", "INTEGER")
]
citation = (
    "Bradley Efron, Trevor Hastie, Iain Johnstone and "
    "Robert Tibshirani (2004) (Least Angle Regression) "
    "Annals of Statistics (with discussion), 407-499"
)
paper_url = "https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf"

# %% [markdown]
# # Create the dataset object
# The definition of all fields can be found in the XSD files describing the
# expected format:
#
# https://github.com/openml/OpenML/blob/master/openml_OS/views/pages/api_new/v1/xsd/openml.data.upload.xsd

#  %%
diabetes_dataset = create_dataset(
    # The name of the dataset (needs to be unique).
    # Must not be longer than 128 characters and only contain
    # a-z, A-Z, 0-9 and the following special characters: _\-\.(),
    name=name,
    # Textual description of the dataset.
    description=description,
    # The person who created the dataset.
    creator="Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani",
    # People who contributed to the current version of the dataset.
    contributor=None,
    # The date the data was originally collected, given by the uploader.
    collection_date="09-01-2012",
    # Language in which the data is represented.
    # Starts with 1 upper case letter, rest lower case, e.g. 'English'.
    language="English",
    # License under which the data is/will be distributed.
    licence="BSD (from scikit-learn)",
    # Name of the target. Can also have multiple values (comma-separated).
    default_target_attribute="class",
    # The attribute that represents the row-id column, if present in the
    # dataset.
    row_id_attribute=None,
    # Attribute or list of attributes that should be excluded in modelling, such as
    # identifiers and indexes. E.g. "feat1" or ["feat1","feat2"]
    ignore_attribute=None,
    # How to cite the paper.
    citation=citation,
    # Attributes of the data
    attributes=attributes,
    data=data,
    # A version label which is provided by the user.
    version_label="test",
    original_data_url="https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html",
    paper_url=paper_url,
)

# %%

diabetes_dataset.publish()
print(f"URL for dataset: {diabetes_dataset.openml_url}")

# %% [markdown]
# ## Dataset is a list
# A list can contain lists in the case of dense data or it can contain
# OrderedDicts in the case of sparse data.
#
# Weather dataset:
# https://storm.cis.fordham.edu/~gweiss/data-mining/datasets.html

# %%
data = [
    ["sunny", 85, 85, "FALSE", "no"],
    ["sunny", 80, 90, "TRUE", "no"],
    ["overcast", 83, 86, "FALSE", "yes"],
    ["rainy", 70, 96, "FALSE", "yes"],
    ["rainy", 68, 80, "FALSE", "yes"],
    ["rainy", 65, 70, "TRUE", "no"],
    ["overcast", 64, 65, "TRUE", "yes"],
    ["sunny", 72, 95, "FALSE", "no"],
    ["sunny", 69, 70, "FALSE", "yes"],
    ["rainy", 75, 80, "FALSE", "yes"],
    ["sunny", 75, 70, "TRUE", "yes"],
    ["overcast", 72, 90, "TRUE", "yes"],
    ["overcast", 81, 75, "FALSE", "yes"],
    ["rainy", 71, 91, "TRUE", "no"],
]

attribute_names = [
    ("outlook", ["sunny", "overcast", "rainy"]),
    ("temperature", "REAL"),
    ("humidity", "REAL"),
    ("windy", ["TRUE", "FALSE"]),
    ("play", ["yes", "no"]),
]

description = (
    "The weather problem is a tiny dataset that we will use repeatedly"
    " to illustrate machine learning methods. Entirely fictitious, it "
    "supposedly concerns the conditions that are suitable for playing "
    "some unspecified game. In general, instances in a dataset are "
    "characterized by the values of features, or attributes, that measure "
    "different aspects of the instance. In this case there are four "
    "attributes: outlook, temperature, humidity, and windy. "
    "The outcome is whether to play or not."
)

citation = (
    "I. H. Witten, E. Frank, M. A. Hall, and ITPro,"
    "Data mining practical machine learning tools and techniques, "
    "third edition. Burlington, Mass.: Morgan Kaufmann Publishers, 2011"
)

weather_dataset = create_dataset(
    name="Weather",
    description=description,
    creator="I. H. Witten, E. Frank, M. A. Hall, and ITPro",
    contributor=None,
    collection_date="01-01-2011",
    language="English",
    licence=None,
    default_target_attribute="play",
    row_id_attribute=None,
    ignore_attribute=None,
    citation=citation,
    attributes=attribute_names,
    data=data,
    version_label="example",
)


# %%
weather_dataset.publish()
print(f"URL for dataset: {weather_dataset.openml_url}")

# %% [markdown]
# ## Dataset is a pandas DataFrame
# It might happen that your dataset is made of heterogeneous data which can usually
# be stored as a Pandas DataFrame. DataFrames offer the advantage of
# storing the type of data for each column as well as the attribute names.
# Therefore, when providing a Pandas DataFrame, OpenML can infer this
# information without needing to explicitly provide it when calling the
# function :func:`openml.datasets.create_dataset`. In this regard, you only
# need to pass ``'auto'`` to the ``attributes`` parameter.

# %%
df = pd.DataFrame(data, columns=[col_name for col_name, _ in attribute_names])

# enforce the categorical column to have a categorical dtype
df["outlook"] = df["outlook"].astype("category")
df["windy"] = df["windy"].astype("bool")
df["play"] = df["play"].astype("category")
print(df.info())

# %% [markdown]
# We enforce the column 'outlook' and 'play' to be a categorical
# dtype while the column 'windy' is kept as a boolean column. 'temperature'
# and 'humidity' are kept as numeric columns. Then, we can
# call :func:`openml.datasets.create_dataset` by passing the dataframe and
# fixing the parameter ``attributes`` to ``'auto'``.

# %%
weather_dataset = create_dataset(
    name="Weather",
    description=description,
    creator="I. H. Witten, E. Frank, M. A. Hall, and ITPro",
    contributor=None,
    collection_date="01-01-2011",
    language="English",
    licence=None,
    default_target_attribute="play",
    row_id_attribute=None,
    ignore_attribute=None,
    citation=citation,
    attributes="auto",
    data=df,
    version_label="example",
)

# %%
weather_dataset.publish()
print(f"URL for dataset: {weather_dataset.openml_url}")

# %% [markdown]
# Dataset is a sparse matrix
# ==========================

# %%
sparse_data = coo_matrix(
    ([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ([0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1]))
)

column_names = [
    ("input1", "REAL"),
    ("input2", "REAL"),
    ("y", "REAL"),
]

xor_dataset = create_dataset(
    name="XOR",
    description="Dataset representing the XOR operation",
    creator=None,
    contributor=None,
    collection_date=None,
    language="English",
    licence=None,
    default_target_attribute="y",
    row_id_attribute=None,
    ignore_attribute=None,
    citation=None,
    attributes=column_names,
    data=sparse_data,
    version_label="example",
)


# %%
xor_dataset.publish()
print(f"URL for dataset: {xor_dataset.openml_url}")


# %% [markdown]
# ## Dataset is a pandas dataframe with sparse columns

sparse_data = coo_matrix(
    ([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], ([0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1]))
)
column_names = ["input1", "input2", "y"]
df = pd.DataFrame.sparse.from_spmatrix(sparse_data, columns=column_names)
print(df.info())

xor_dataset = create_dataset(
    name="XOR",
    description="Dataset representing the XOR operation",
    creator=None,
    contributor=None,
    collection_date=None,
    language="English",
    licence=None,
    default_target_attribute="y",
    row_id_attribute=None,
    ignore_attribute=None,
    citation=None,
    attributes="auto",
    data=df,
    version_label="example",
)

# %%

xor_dataset.publish()
print(f"URL for dataset: {xor_dataset.openml_url}")

# %%
openml.config.stop_using_configuration_for_example()
# License: BSD 3-Clause
