:orphan:

.. _progress:

========
Progress
========

API calls
=========

=============================================== =========== ====== =============== ========== =====================
API call                                        implemented tested properly tested loads json proper error handling
=============================================== =========== ====== =============== ========== =====================
/data/list/                                     yes         yes
/data/list/tag/{tag}
/data/{data_id}                                 yes         yes
/data/delete/
/data/upload/
/data/features/{data_id}                        yes         yes
/data/features/upload/
/data/qualities/{data_id}                       yes         yes
/data/qualities/list
/data/qualities/upload
/data/tag
/data/untag
/task/list                                      yes         yes
/task/list/tag/{tag}
/task/{task_id}                                 yes         yes
/task/tag
/task/untag
/task/delete
/tasktype/list
/tasktype/{task_id}
/flow/list                                      yes
/flow/tag
/flow/untag
/flow/{flow_id}
/flow/
/flow/exists/{name,ext_version}
/flow/owned
/run/list                                       yes         yes
/run/{run_id}                                   yes         yes
/run
/run/tag
/run/untag
/run/evaluate
/run/reset
/estimationprocedure/{proc_id}
/estimationprocedure/list
/evaluationmeasures/list
/job/request/
=============================================== =========== ====== =============== ========== =====================

This list does not contain the `/setup/` calls because we do not need them
according to Jan.

Convenience Functions
=====================

=============================================== =========== ====== =============== ========== =====================
Method                                          implemented tested properly tested loads json proper error handling
=============================================== =========== ====== =============== ========== =====================
get_cached_split                                yes
get_cached_splits                               yes
get_cached_dataset                              yes         yes
get_cached_datasets                             yes         yes
get_cached_task                                 yes
get_cached_tasks                                yes
=============================================== =========== ====== =============== ========== =====================
