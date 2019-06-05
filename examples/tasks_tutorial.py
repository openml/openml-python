"""
Tasks
=====

A tutorial on how to list and download tasks.
"""

import openml
import pandas as pd
from pprint import pprint

############################################################################
#
# Tasks are identified by IDs and can be accessed in two different ways:
#
# 1. In a list providing basic information on all tasks available on OpenML.
# This function will not download the actual tasks, but will instead download
# meta data that can be used to filter the tasks and retrieve a set of IDs.
# We can filter this list, for example, we can only list tasks having a
# special tag or only tasks for a specific target such as
# *supervised classification*.
#
# 2. A single task by its ID. It contains all meta information, the target
# metric, the splits and an iterator which can be used to access the
# splits in a useful manner.

############################################################################
# Listing tasks
# ^^^^^^^^^^^^^
#
# We will start by simply listing only *supervised classification* tasks:

tasks = openml.tasks.list_tasks(task_type_id=1)

############################################################################
# **openml.tasks.list_tasks()** returns a dictionary of dictionaries, we convert it into a
# `pandas dataframe <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
# to have better visualization and easier access:

tasks = pd.DataFrame.from_dict(tasks, orient='index')
print(tasks.columns)
print("First 5 of %s tasks:" % len(tasks))
pprint(tasks.head())

# The same can be obtained through lesser lines of code
tasks_df = openml.tasks.list_tasks(task_type_id=1, output_format='dataframe')
pprint(tasks_df.head())

############################################################################
# We can filter the list of tasks to only contain datasets with more than
# 500 samples, but less than 1000 samples:

filtered_tasks = tasks.query('NumberOfInstances > 500 and NumberOfInstances < 1000')
print(list(filtered_tasks.index))

############################################################################

# Number of tasks
print(len(filtered_tasks))

############################################################################
# Then, we can further restrict the tasks to all have the same resampling strategy:

filtered_tasks = filtered_tasks.query('estimation_procedure == "10-fold Crossvalidation"')
print(list(filtered_tasks.index))

############################################################################

# Number of tasks
print(len(filtered_tasks))

############################################################################
# Resampling strategies can be found on the
# `OpenML Website <http://www.openml.org/search?type=measure&q=estimation%20procedure>`_.
#
# Similar to listing tasks by task type, we can list tasks by tags:

tasks = openml.tasks.list_tasks(tag='OpenML100')
tasks = pd.DataFrame.from_dict(tasks, orient='index')
print("First 5 of %s tasks:" % len(tasks))
pprint(tasks.head())

############################################################################
# Furthermore, we can list tasks based on the dataset id:

tasks = openml.tasks.list_tasks(data_id=1471)
tasks = pd.DataFrame.from_dict(tasks, orient='index')
print("First 5 of %s tasks:" % len(tasks))
pprint(tasks.head())

############################################################################
# In addition, a size limit and an offset can be applied both separately and simultaneously:

tasks = openml.tasks.list_tasks(size=10, offset=50)
tasks = pd.DataFrame.from_dict(tasks, orient='index')
pprint(tasks)

############################################################################
#
# **OpenML 100**
# is a curated list of 100 tasks to start using OpenML. They are all
# supervised classification tasks with more than 500 instances and less than 50000
# instances per task. To make things easier, the tasks do not contain highly
# unbalanced data and sparse data. However, the tasks include missing values and
# categorical features. You can find out more about the *OpenML 100* on
# `the OpenML benchmarking page <https://www.openml.org/guide/benchmark>`_.
#
# Finally, it is also possible to list all tasks on OpenML with:

############################################################################
tasks = openml.tasks.list_tasks()
tasks = pd.DataFrame.from_dict(tasks, orient='index')
print(len(tasks))

############################################################################
# Exercise
# ########
#
# Search for the tasks on the 'eeg-eye-state' dataset.

tasks.query('name=="eeg-eye-state"')

############################################################################
# Downloading tasks
# ^^^^^^^^^^^^^^^^^
#
# We provide two functions to download tasks, one which downloads only a
# single task by its ID, and one which takes a list of IDs and downloads
# all of these tasks:

task_id = 31
task = openml.tasks.get_task(task_id)

############################################################################
# Properties of the task are stored as member variables:

pprint(vars(task))

############################################################################
# And:

ids = [2, 1891, 31, 9983]
tasks = openml.tasks.get_tasks(ids)
pprint(tasks[0])
