:orphan:

.. _progress:

=========
Changelog
=========

0.9.0
~~~~~
* ADD #560: OpenML-Python can now handle regression tasks as well.
* ADD #620, #628, #632, #649, #682: Full support for studies and distinguishes suites from studies.
* ADD #607: Tasks can now be created and uploaded.
* ADD #647, #673: Introduced the extension interface. This provides an easy way to create a hook for machine learning packages to perform e.g. automated runs.
* ADD #548, #646, #676: Support for Pandas DataFrame and SparseDataFrame
* ADD #662: Results of listing functions can now be returned as pandas.DataFrame.
* ADD #59: Datasets can now also be retrieved by name.
* ADD #672: Add timing measurements for runs, when possible.
* ADD #661: Upload time and error messages now displayed with `list_runs`.
* ADD #644: Datasets can now be downloaded 'lazily', retrieving only metadata at first, and the full dataset only when necessary.
* ADD #659: Lazy loading of task splits.
* ADD #516: `run_flow_on_task` flow uploading is now optional.
* ADD #680: Adds `openml.config.start_using_configuration_for_example` (and resp. stop) to easily connect to the test server.
* FIX #642: `check_datasets_active` now correctly also returns active status of deactivated datasets.
* FIX #304, #636: Allow serialization of numpy datatypes and list of lists of more types (e.g. bools, ints) for flows.
* FIX #651: Fixed a bug that would prevent openml-python from finding the user's config file.
* DOC #678: Better color scheme for code examples in documentation.
* DOC #681: Small improvements and removing list of missing functions.
* DOC #684: Add notice to examples that connect to the test server.
* DOC #691: Update contributing guidelines to use Github draft feature instead of tags in title.
* MAINT #184: Dropping Python2 support.
* MAINT #596: Fewer dependencies for regular pip install.
* MAINT #652: Numpy and Scipy are no longer required before installation.
* MAINT #655: Lazy loading is now preferred in unit tests.
* MAINT #667: Different tag functions now share code.
* MAINT #666: More descriptive error message for `TypeError` in `list_runs`.
* MAINT #668: Fix some type hints.
* MAINT #677: `dataset.get_data` now has consistent behavior in its return type.
* MAINT #686: Adds ignore directives for several `mypy` folders.
* MAINT #629, #630: Code now adheres to single PEP8 standard.

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

* Add this changelog
* 2nd example notebook PyOpenML.ipynb
* Pagination support for list datasets and list tasks

Prior
~~~~~

There is no changelog for prior versions.
