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
print(f"This is dataset '{dataset.name}', the target feature is "
      f"'{dataset.default_target_attribute}'")
print(f"URL: {dataset.url}")
print(dataset.description[:500])
