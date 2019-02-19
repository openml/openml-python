:orphan:

.. _progress:

========
Progress
========

Changelog
=========

0.9.0
~~~~~

* ADD #560: OpenML-Python can now handle regression tasks as well.
* MAINT #184: Dropping Python2 support.

0.8.0
~~~~~

* ADD #440: Improved dataset upload.
* ADD #545, #583: Allow uploading a dataset from a pandas DataFrame.
* ADD #528: New functions to update the status of a dataset.
* ADD #523: Support for scikit-learn 0.20's new ColumnTransformer.
* ADD #459: Enhanced support to store runs on disk prior to uploading them to
  OpenML.
* ADD #564: New helpers to access the structure of a flow (and find its
  subflows).
* ADD #618: The software will from now on retry to connect to the server if a
  connection failed. The number of retries can be configured.
* FIX #538: Support loading clustering tasks.
* FIX #464: Fixes a bug related to listing functions (returns correct listing
  size).
* FIX #580: Listing function now works properly when there are less results
  than requested.
* FIX #571: Fixes an issue where tasks could not be downloaded in parallel.
* FIX #536: Flows can now be printed when the flow name is None.
* FIX #504: Better support for hierarchical hyperparameters when uploading
  scikit-learn's grid and random search.
* FIX #569: Less strict checking of flow dependencies when loading flows.
* FIX #431: Pickle of task splits are no longer cached.
* DOC #540: More examples for dataset uploading.
* DOC #554: Remove the doubled progress entry from the docs.
* MAINT #613: Utilize the latest updates in OpenML evaluation listings.
* MAINT #482: Cleaner interface for handling search traces.
* MAINT #557: Continuous integration works for scikit-learn 0.18-0.20.
* MAINT #542: Continuous integration now runs python3.7 as well.
* MAINT #535: Continuous integration now enforces PEP8 compliance for new code.
* MAINT #527: Replace deprecated nose by pytest.
* MAINT #510: Documentation is now built by travis-ci instead of circle-ci.
* MAINT: Completely re-designed documentation built on sphinx gallery.
* MAINT #462: Appveyor CI support.
* MAINT #477: Improve error handling for issue
  `#479 <https://github.com/openml/openml-python/pull/479>`_:
  the OpenML connector fails earlier and with a better error message when
  failing to create a flow from the OpenML description.
* MAINT #561: Improve documentation on running specific unit tests.

0.4.-0.7
~~~~~~~~

There is no changelog for these versions.

0.3.0
~~~~~

* Add this changelog (Matthias Feurer)
* 2nd example notebook PyOpenML.ipynb (Joaquin Vanschoren)
* Pagination support for list datasets and list tasks

Prior
~~~~~

There is no changelog for prior versions.

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
