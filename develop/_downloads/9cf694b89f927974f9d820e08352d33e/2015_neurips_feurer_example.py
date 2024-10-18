"""
Feurer et al. (2015)
====================

A tutorial on how to get the datasets used in the paper introducing *Auto-sklearn* by Feurer et al..

Auto-sklearn website: https://automl.github.io/auto-sklearn/

Publication
~~~~~~~~~~~

| Efficient and Robust Automated Machine Learning
| Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum and Frank Hutter
| In *Advances in Neural Information Processing Systems 28*, 2015
| Available at https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
"""  # noqa F401

# License: BSD 3-Clause

import pandas as pd

import openml

####################################################################################################
# List of dataset IDs given in the supplementary material of Feurer et al.:
# https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning-supplemental.zip
# fmt: off
dataset_ids = [
    3, 6, 12, 14, 16, 18, 21, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 44, 46,
    57, 60, 179, 180, 181, 182, 184, 185, 273, 293, 300, 351, 354, 357, 389,
    390, 391, 392, 393, 395, 396, 398, 399, 401, 554, 679, 715, 718, 720, 722,
    723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799,
    803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847,
    849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930,
    934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995,
    1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1049, 1050, 1053,
    1056, 1067, 1068, 1069, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130,
    1134, 1138, 1139, 1142, 1146, 1161, 1166,
]
# fmt: on

####################################################################################################
# The dataset IDs could be used directly to load the dataset and split the data into a training set
# and a test set. However, to be reproducible, we will first obtain the respective tasks from
# OpenML, which define both the target feature and the train/test split.
#
# .. note::
#    It is discouraged to work directly on datasets and only provide dataset IDs in a paper as
#    this does not allow reproducibility (unclear splitting). Please do not use datasets but the
#    respective tasks as basis for a paper and publish task IDS. This example is only given to
#    showcase the use of OpenML-Python for a published paper and as a warning on how not to do it.
#    Please check the `OpenML documentation of tasks <https://docs.openml.org/concepts/tasks/>`_ if you
#    want to learn more about them.

####################################################################################################
# This lists both active and inactive tasks (because of ``status='all'``). Unfortunately,
# this is necessary as some of the datasets contain issues found after the publication and became
# deactivated, which also deactivated the tasks on them. More information on active or inactive
# datasets can be found in the `online docs <https://docs.openml.org/concepts/data/#dataset-status>`_.
tasks = openml.tasks.list_tasks(
    task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
    status="all",
    output_format="dataframe",
)

# Query only those with holdout as the resampling startegy.
tasks = tasks.query('estimation_procedure == "33% Holdout set"')

task_ids = []
for did in dataset_ids:
    tasks_ = list(tasks.query("did == {}".format(did)).tid)
    if len(tasks_) >= 1:  # if there are multiple task, take the one with lowest ID (oldest).
        task_id = min(tasks_)
    else:
        raise ValueError(did)

    # Optional - Check that the task has the same target attribute as the
    # dataset default target attribute
    # (disabled for this example as it needs to run fast to be rendered online)
    # task = openml.tasks.get_task(task_id)
    # dataset = task.get_dataset()
    # if task.target_name != dataset.default_target_attribute:
    #     raise ValueError(
    #         (task.target_name, dataset.default_target_attribute)
    #     )

    task_ids.append(task_id)

assert len(task_ids) == 140
task_ids.sort()

# These are the tasks to work with:
print(task_ids)
