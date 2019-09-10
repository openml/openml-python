"""
========
Datasets
========

A basic tutorial on how to list and download datasets.
"""
############################################################################
import openml

############################################################################
# List datasets
# =============

datasets_df = openml.datasets.list_datasets(output_format='dataframe')
print(datasets_df.head(n=10))

############################################################################
# Download a dataset
# ==================

first_dataset_id = int(datasets_df['did'].iloc[0])
dataset = openml.datasets.get_dataset(first_dataset_id)

# Print a summary
print("This is dataset '%s', the target feature is '%s'" %
      (dataset.name, dataset.default_target_attribute))
print("URL: %s" % dataset.url)
print(dataset.description[:500])
