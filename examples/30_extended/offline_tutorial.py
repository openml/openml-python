"""
======================
Working Offline
======================

This tutorial explains how to use OpenML-Python without an active
internet connection by pre-populating the local cache.

OpenML-Python stores downloaded data locally so that subsequent
calls can be served from the cache without contacting the server.
"""

# %%
# .. note::
#     You need an internet connection to populate the cache initially.
#     Once populated, all operations below work fully offline.

# %%
# Setting up your API key
# =======================
# Before using OpenML, configure your API key:

import openml

# Set your API key (only needed for online step)
# openml.config.apikey = "YOUR_API_KEY_HERE"

# %%
# Step 1 - Populate the cache (requires internet)
# ================================================
# Use 'populate_cache' to download and store datasets, tasks,
# flows, and runs locally before going offline.

openml.populate_cache(
    task_ids=[31, 37],       # Download specific tasks
    dataset_ids=[31, 40],    # Download specific datasets
)

print("Cache populated successfully. You can now work offline.")

# %%
# Step 2 - Check your local cache directory
# ==========================================
# OpenML stores cached data in a local directory.
# You can find and configure it as follows:

cache_dir = openml.config.get_cache_directory()
print(f"Cache directory: {cache_dir}")

#%%
# Step 3 - Load a dataset from cache (no internet needed)
# ========================================================
# Once cached, datasets load directly from disk:

dataset = openml.datasets.get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)

print(f"Dataset: {dataset.name}")
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

# %%
# Step 4 - Load a task from cache (no internet needed)
# =====================================================

task = openml.tasks.get_task(31)
print(f"Task type: {task.task_type}")
print(f"Target: {task.target_name}")

# %%
# Step 5 - Run an experiment offline
# ===================================
# You can run experiments using cached data without internet access:

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=3, random_state=42)

train_indices, test_indices = task.get_train_test_split_indices(
    fold=0,
    repeat=0,
)

X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")

# %%
# Summary
# =======
# Key points for working offline with OpenML-Python:
#
# 1. Use 'openml.populate_cache()' to download data before going offline
# 2. Cached data is stored in 'openml.config.get_cache_directory()'
# 3. Once cached, 'get_dataset()' and 'get_task()' work without internet
# 4. Re-run 'populate_cache()' periodically to refresh stale data