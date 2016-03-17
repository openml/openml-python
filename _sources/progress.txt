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
/data/upload/                                   yes         yes
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
/flow/                                          yes         yes
/flow/exists/{name,ext_version}
/flow/owned
/run/list/task/{task_ids}                       yes         yes
/run/list/run/{run_ids}                         yes         yes
/run/list/tag/{tag}                             yes         yes
/run/{id}                                       yes         yes
/run/list/uploader/{ids}                        yes         yes
/run/list/flow/{ids}                            yes         yes
/run/list/{filters}                             yes         yes
/run/tag
/run/untag
/run (POST)                                     yes         yes
/run/{id} (DELETE)
/estimationprocedure/{proc_id}
/estimationprocedure/list
/evaluationmeasures/list
/job/request/
=============================================== =========== ====== =============== ========== =====================

Convenience Functions
=====================

=============================================== =========== ====== =============== ========== =====================
Method                                          implemented tested properly tested loads json proper error handling
=============================================== =========== ====== =============== ========== =====================
_get_cached_split                               yes
_get_cached_splits                              yes
_get_cached_dataset                             yes         yes
_get_cached_datasets                            yes         yes
get_cached_task                                 yes
get_cached_tasks                                yes
=============================================== =========== ====== =============== ========== =====================
