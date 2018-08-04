"""
Tasks
=====

A tutorial on how to list and download tasks.
"""

import openml
import pandas as pd
from pprint import pprint

############################################################################
# **Set your own goals and invite others to work on the same problem**
#
# Note: tasks are typically created in the web interface
#
# **Listing tasks**

# we are going to pull
task_list = openml.tasks.list_tasks(size=1000)  # Get first 1000 tasks
mytasks = pd.DataFrame.from_dict(task_list, orient='index')
mytasks = mytasks[['tid', 'did', 'name', 'task_type',
                   'estimation_procedure']]
print("First 5 of %s tasks:" % len(mytasks))
mytasks.head()

############################################################################
# **Exercise**
#
# * Search for the tasks on the 'eeg-eye-state' dataset.
mytasks.query('name=="eeg-eye-state"')

############################################################################
# **Download tasks**

task = openml.tasks.get_task(1)
pprint(vars(task))
