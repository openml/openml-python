"""
=====
Tasks
=====

A 20_basic tutorial on how to list and download tasks.
"""
import openml
from pprint import pprint
############################################################################
# Listing tasks
# =============
#
# Listing only supervised classification tasks (task type 1)
# from all `task types <https://www.openml.org/search?type=task_type>`_.

tasks_df = openml.tasks.list_tasks(task_type_id=1, output_format='dataframe')
pprint(tasks_df.head())

############################################################################
# Downloading a task
# ==================

first_task_id = int(tasks_df['tid'].iloc[0])
task = openml.tasks.get_task(first_task_id)
pprint(task)
