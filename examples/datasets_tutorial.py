"""
Datasets
========

How to list and download datasets.
"""

import openml
import pandas as pd

############################################################################
# List datasets
# ^^^^^^^^^^^^^

openml_list = openml.datasets.list_datasets()  # returns a dict

# Show a nice table with some key data properties
datalist = pd.DataFrame.from_dict(openml_list, orient='index')
datalist = datalist[[
    'did', 'name', 'NumberOfInstances',
    'NumberOfFeatures', 'NumberOfClasses'
]]

print("First 10 of %s datasets..." % len(datalist))
datalist.head(n=10)

############################################################################

# Exercise
# ********
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
# ^^^^^^^^^^^^^^^^^

# This is done based on the dataset ID ('did').
dataset = openml.datasets.get_dataset(68)

# Print a summary
print("This is dataset '%s', the target feature is '%s'" %
      (dataset.name, dataset.default_target_attribute))
print("URL: %s" % dataset.url)
print(dataset.description[:500])

############################################################################
# Get the actual data.
#
# Returned as numpy array, with meta-info (e.g. target feature, feature names,...)
X, y, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    return_attribute_names=True,
)
eeg = pd.DataFrame(X, columns=attribute_names)
eeg['class'] = y
print(eeg[:10])

############################################################################

# Exercise
# ********
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