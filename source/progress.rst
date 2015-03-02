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
authenticate                                    yes         yes
authenticate.check
data                                            yes         yes
data.description                                yes         yes
data.upload
data.delete
data.licences
data.features                                   yes         yes
data.qualities                                  yes         yes
data.qualities.list
task                                            yes         yes
task.types.search                               yes         yes
task.evaluations
task.types
estimationprocedure
implementation.exists
implementation.upload
implementation.owned
implementation.delete
implementation.licences
evaluation.measures
run
run.upload
run.delete
job
setup
=============================================== =========== ====== =============== ========== =====================

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
