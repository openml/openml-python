:orphan:

.. _progress:

========
Progress
========

Changelog
=========

0.8.0
~~~~~
* Added serialize run / deserialize run function (for saving runs on disk before uploading)
* FIX: fixed bug related to listing functions (returns correct listing size)
* made openml.utils.list_all a hidden function (should be accessed only by the respective listing functions)
* Improve error handling for issue `#479 <https://github.com/openml/openml-python/pull/479>`_:
  the OpenML connector fails earlier and with a better error message when
  failing to create a flow from the OpenML description.

0.3.0
~~~~~

* Add this changelog (Matthias Feurer)
* 2nd example notebook PyOpenML.ipynb (Joaquin Vanschoren)
* Pagination support for list datasets and list tasks

API calls
=========

=============================================== =========== ====== =============== ========== =====================
API call                                        implemented tested properly tested loads json proper error handling
=============================================== =========== ====== =============== ========== =====================
/data/{id}                                      yes         yes
/data/features/{id}                             yes         yes
/data/qualities/{id}                            yes         yes
/data/list/                                     yes         yes
/data/list/tag/{tag}                            yes         yes
/data/upload/                                   yes         yes
/data/tag
/data/untag
/data/delete/                                   X

/task/{task}                                    yes         yes
/task/list                                      yes         yes
/task/list/type/{id}                            yes         yes
/task/list/tag/{tag}                            yes         yes
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
