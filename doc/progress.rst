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
/data/{id}                                      yes         yes
/data/features/{id}                             yes         yes
/data/qualities/{id}                            yes         yes
/data/list/                                     yes         yes
/data/list/tag/{tag}
/data/upload/                                   yes         yes
/data/tag
/data/untag
/data/delete/                                   X

/task/{task}                                    yes         yes
/task/list                                      yes         yes
/task/list/type/{id}
/task/list/tag/{tag}
/task {POST}
/task/tag
/task/untag
/task/delete                                    X

/tasktype/{id}
/tasktype/list

/flow/{id}
/flow/exists/{name}/{ext_version}               yes
/flow/list                                      yes
/flow/list/tag/{tag}
/flow/owned
/flow/ {POST}                                   yes         yes
/flow/tag
/flow/untag
/flow/{id} {DELETE}                             X

/run/list/task/{ids}                            yes         yes
/run/list/run/{ids}                             yes         yes
/run/list/tag/{tag}                             yes         yes
/run/{id}                                       yes         yes
/run/list/uploader/{ids}                        yes         yes
/run/list/flow/{ids}                            yes         yes
/run/list/{filters}                             yes         yes
/run/untag
/run (POST)                                     yes         yes
/run/tag
/run/{id} (DELETE)                              X

/evaluation/list/run/{ids}
/evaluation/list/tag/{tag}
/evaluation/list/task/{ids}
/evaluation/list/uploader/{ids}
/evaluation/list/flow/{ids}
/evaluation/list/{filters}

=============================================== =========== ====== =============== ========== =====================

We do not plan to implement API calls marked with an **X**!

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
