"""
========================================
Benchmarking with Progress Tracking
========================================

This tutorial demonstrates how to benchmark machine learning models
across entire OpenML suites with real-time progress tracking.
"""

# License: BSD 3-Clause

# %% [markdown]
# ## Introduction
#
# The ``run_suite_with_progress`` function provides an easy way to benchmark
# models across entire OpenML benchmark suites with real-time progress tracking
# via the tqdm library.

# %%
import openml
import openml_sklearn  # Required for sklearn models
from sklearn.ensemble import RandomForestClassifier

from openml.study import run_suite_with_progress

# Configure to use test server for examples
openml.config.start_using_configuration_for_example()

# %% [markdown]
# .. note::
#     This example requires:
#
#     * ``pip install tqdm`` for progress bars
#     * ``pip install openml-sklearn`` for sklearn model support
#     * OpenML API key - get from https://test.openml.org (test server)
#       or https://www.openml.org (main server)
#
#     Set your API key with:
#
#     >>> openml.config.apikey = 'YOURKEY'

# %% [markdown]
# ## Running a Benchmark with Progress Bar
#
# The progress bar shows real-time updates as tasks complete. When running,
# you'll see something like:
#
# .. code-block:: text
#
#     Benchmarking Test Suite: 100%|████████| 3/3 [01:23<00:00, 27.8s/task]

# %%
# Create a simple model
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)

# Run benchmark on a test suite with progress bar
# Note: This will fail without an API key
try:
    runs = run_suite_with_progress(
        suite_id=99,  # Test suite ID on test server
        model=clf,
        show_progress=True,  # Shows progress bar (requires tqdm)
    )
    print(f"\nCompleted {len(runs)} tasks successfully!")
except openml.exceptions.OpenMLServerException as e:
    print(f"Server error (likely missing API key): {e}")
    print("Set your API key with: openml.config.apikey = 'YOURKEY'")

# %% [markdown]
# ## Running Without Progress Bar
#
# You can disable the progress bar if tqdm is not installed, or if you
# prefer cleaner output in log files.

# %%
try:
    runs = run_suite_with_progress(
        suite_id=99,
        model=clf,
        show_progress=False,  # No progress bar
    )
    print(f"Completed {len(runs)} tasks")
except openml.exceptions.OpenMLServerException as e:
    print(f"Server error: {e}")

# %% [markdown]
# ## Passing Additional Parameters
#
# All parameters from ``run_model_on_task`` are supported via ``**run_kwargs``:

# %%
try:
    runs = run_suite_with_progress(
        suite_id=99,
        model=clf,
        show_progress=True,
        # Additional run_model_on_task parameters:
        avoid_duplicate_runs=False,  # Allow re-running tasks
        upload_flow=True,  # Upload the model flow
    )
except openml.exceptions.OpenMLServerException as e:
    print(f"Server error: {e}")

# %% [markdown]
# ## Working with Results
#
# The function returns a list of ``OpenMLRun`` objects that can be analyzed:

# %%
# Example of working with results (when runs are successful)
# for run in runs:
#     print(f"Task {run.task_id}: Run ID {run.run_id}")
#     # Access evaluations
#     if run.evaluations:
#         for metric, value in run.evaluations.items():
#             print(f"  {metric}: {value}")

# %% [markdown]
# ## Common Use Cases
#
# **Quick Testing on Small Suites:**
#
# .. code-block:: python
#
#     import openml
#     import openml_sklearn
#     from sklearn.ensemble import RandomForestClassifier
#     from openml.study import run_suite_with_progress
#
#     clf = RandomForestClassifier(random_state=42)
#     # Suite 334: OpenML-Tiny (7 tasks) - on main server
#     runs = run_suite_with_progress(suite_id=334, model=clf)
#
# **Production Benchmarking:**
#
# .. code-block:: python
#
#     # Suite 99: OpenML-CC18 (72 classification tasks) - on main server
#     runs = run_suite_with_progress(suite_id=99, model=clf)
#
# **Comparing Multiple Models:**
#
# .. code-block:: python
#
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.tree import DecisionTreeClassifier
#
#     models = {
#         'RandomForest': RandomForestClassifier(random_state=42),
#         'LogisticRegression': LogisticRegression(random_state=42),
#         'DecisionTree': DecisionTreeClassifier(random_state=42),
#     }
#
#     results = {}
#     for name, model in models.items():
#         print(f"Benchmarking {name}...")
#         results[name] = run_suite_with_progress(suite_id=334, model=model)

# %% [markdown]
# ## Requirements
#
# * ``tqdm`` package for progress bars: ``pip install tqdm``
# * ``openml-sklearn`` extension: ``pip install openml-sklearn``
# * OpenML API key (get from https://www.openml.org/auth/api-key)
# * Internet connection to access OpenML server
#
# To disable progress bar if tqdm is not available, set ``show_progress=False``

# %%
openml.config.stop_using_configuration_for_example()
