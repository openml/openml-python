"""
========
Datasets
========

How to list and download datasets.
"""
############################################################################
import openml
import pandas as pd

############################################################################
# Exercise 0
# **********
#
# * List datasets
#   * Use the output_format parameter to select output type
#   * Default gives 'dict' (other option: 'dataframe')

openml_list = openml.datasets.list_datasets()  # returns a dict

# Show a nice table with some key data properties
datalist = pd.DataFrame.from_dict(openml_list, orient='index')
datalist = datalist[[
    'did', 'name', 'NumberOfInstances',
    'NumberOfFeatures', 'NumberOfClasses'
]]

print("First 10 of %s datasets..." % len(datalist))
datalist.head(n=10)

# The same can be done with lesser lines of code
openml_df = openml.datasets.list_datasets(output_format='dataframe')
openml_df.head(n=10)

############################################################################
# Exercise 1
# **********
#
# * Find datasets with more than 10000 examples.
# * Find a dataset called 'eeg_eye_state'.
# * Find all datasets with more than 50 classes.
datalist[datalist.NumberOfInstances > 10000
         ].sort_values(['NumberOfInstances']).head(n=20)
############################################################################
datalist.query('name == "eeg-eye-state"')
############################################################################
datalist.query('NumberOfClasses > 50')

############################################################################
# Download datasets
# =================

# This is done based on the dataset ID.
dataset = openml.datasets.get_dataset(1471)

# Print a summary
print("This is dataset '%s', the target feature is '%s'" %
      (dataset.name, dataset.default_target_attribute))
print("URL: %s" % dataset.url)
print(dataset.description[:500])

############################################################################
# Get the actual data.
#
# The dataset can be returned in 2 possible formats: as a NumPy array, a SciPy
# sparse matrix, or as a Pandas DataFrame (or SparseDataFrame). The format is
# controlled with the parameter ``dataset_format`` which can be either 'array'
# (default) or 'dataframe'. Let's first build our dataset from a NumPy array
# and manually create a dataframe.
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)
eeg = pd.DataFrame(X, columns=attribute_names)
eeg['class'] = y
print(eeg[:10])

############################################################################
# Instead of manually creating the dataframe, you can already request a
# dataframe with the correct dtypes.
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format='dataframe'
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
    c=eegs[:100]['class'],
    figsize=(10, 10),
    marker='o',
    hist_kwds={'bins': 20},
    alpha=.8,
    cmap='plasma'
)
