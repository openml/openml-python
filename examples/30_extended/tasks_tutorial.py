"""
Tasks
=====

A tutorial on how to list and download tasks.
"""

# License: BSD 3-Clause

import openml
from openml.tasks import TaskType
import pandas as pd

############################################################################
#
# Tasks are identified by IDs and can be accessed in two different ways:
#
# 1. In a list providing basic information on all tasks available on OpenML.
#    This function will not download the actual tasks, but will instead download
#    meta data that can be used to filter the tasks and retrieve a set of IDs.
#    We can filter this list, for example, we can only list tasks having a
#    special tag or only tasks for a specific target such as
#    *supervised classification*.
# 2. A single task by its ID. It contains all meta information, the target
#    metric, the splits and an iterator which can be used to access the
#    splits in a useful manner.

############################################################################
# Listing tasks
# ^^^^^^^^^^^^^
#
# We will start by simply listing only *supervised classification* tasks.
# **openml.tasks.list_tasks()** returns a dictionary of dictionaries by default, but we
# request a
# `pandas dataframe <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
# instead to have better visualization capabilities and easier access:

tasks = openml.tasks.list_tasks(
    task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"
)
print(tasks.columns)
print(f"First 5 of {len(tasks)} tasks:")
print(tasks.head())

############################################################################
# We can filter the list of tasks to only contain datasets with more than
# 500 samples, but less than 1000 samples:

filtered_tasks = tasks.query("NumberOfInstances > 500 and NumberOfInstances < 1000")
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
# `OpenML Website <https://www.openml.org/search?type=measure&q=estimation%20procedure>`_.
#
# Similar to listing tasks by task type, we can list tasks by tags:

tasks = openml.tasks.list_tasks(tag="OpenML100", output_format="dataframe")
print(f"First 5 of {len(tasks)} tasks:")
print(tasks.head())

############################################################################
# Furthermore, we can list tasks based on the dataset id:

tasks = openml.tasks.list_tasks(data_id=1471, output_format="dataframe")
print(f"First 5 of {len(tasks)} tasks:")
print(tasks.head())

############################################################################
# In addition, a size limit and an offset can be applied both separately and simultaneously:

tasks = openml.tasks.list_tasks(size=10, offset=50, output_format="dataframe")
print(tasks)

############################################################################
#
# **OpenML 100**
# is a curated list of 100 tasks to start using OpenML. They are all
# supervised classification tasks with more than 500 instances and less than 50000
# instances per task. To make things easier, the tasks do not contain highly
# unbalanced data and sparse data. However, the tasks include missing values and
# categorical features. You can find out more about the *OpenML 100* on
# `the OpenML benchmarking page <https://docs.openml.org/benchmark/>`_.
#
# Finally, it is also possible to list all tasks on OpenML with:

############################################################################
tasks = openml.tasks.list_tasks(output_format="dataframe")
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

print(task)

############################################################################
# And:

ids = [2, 1891, 31, 9983]
tasks = openml.tasks.get_tasks(ids)
print(tasks[0])

############################################################################
# Creating tasks
# ^^^^^^^^^^^^^^
#
# You can also create new tasks. Take the following into account:
#
# * You can only create tasks on *active* datasets
# * For now, only the following tasks are supported: classification, regression,
#   clustering, and learning curve analysis.
# * For now, tasks can only be created on a single dataset.
# * The exact same task must not already exist.
#
# Creating a task requires the following input:
#
# * task_type: The task type ID, required (see below). Required.
# * dataset_id: The dataset ID. Required.
# * target_name: The name of the attribute you aim to predict. Optional.
# * estimation_procedure_id : The ID of the estimation procedure used to create train-test
#   splits. Optional.
# * evaluation_measure: The name of the evaluation measure. Optional.
# * Any additional inputs for specific tasks
#
# It is best to leave the evaluation measure open if there is no strong prerequisite for a
# specific measure. OpenML will always compute all appropriate measures and you can filter
# or sort results on your favourite measure afterwards. Only add an evaluation measure if
# necessary (e.g. when other measure make no sense), since it will create a new task, which
# scatters results across tasks.

############################################################################
# We'll use the test server for the rest of this tutorial.
#
# .. warning::
#    .. include:: ../../test_server_usage_warning.txt
openml.config.start_using_configuration_for_example()

############################################################################
# Example
# #######
#
# Let's create a classification task on a dataset. In this example we will do this on the
# Iris dataset (ID=128 (on test server)). We'll use 10-fold cross-validation (ID=1),
# and *predictive accuracy* as the predefined measure (this can also be left open).
# If a task with these parameters exists, we will get an appropriate exception.
# If such a task doesn't exist, a task will be created and the corresponding task_id
# will be returned.


try:
    my_task = openml.tasks.create_task(
        task_type=TaskType.SUPERVISED_CLASSIFICATION,
        dataset_id=128,
        target_name="class",
        evaluation_measure="predictive_accuracy",
        estimation_procedure_id=1,
    )
    my_task.publish()
except openml.exceptions.OpenMLServerException as e:
    # Error code for 'task already exists'
    if e.code == 614:
        # Lookup task
        tasks = openml.tasks.list_tasks(data_id=128, output_format="dataframe")
        tasks = tasks.query(
            'task_type == "Supervised Classification" '
            'and estimation_procedure == "10-fold Crossvalidation" '
            'and evaluation_measures == "predictive_accuracy"'
        )
        task_id = tasks.loc[:, "tid"].values[0]
        print("Task already exists. Task ID is", task_id)

# reverting to prod server
openml.config.stop_using_configuration_for_example()


############################################################################
# * `Complete list of task types <https://www.openml.org/search?type=task_type>`_.
# * `Complete list of model estimation procedures <https://www.openml.org/search?q=%2520measure_type%3Aestimation_procedure&type=measure>`_.
# * `Complete list of evaluation measures <https://www.openml.org/search?q=measure_type%3Aevaluation_measure&type=measure>`_.
#
