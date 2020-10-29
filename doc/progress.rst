:orphan:

.. _progress:

=========
Changelog
=========

0.11.1
~~~~~~
* MAINT #577: Added unit test to verify how the dataset object handles comparisons.
* MAINT #671: Improved the performance of ``check_datasets_active`` by only querying the given list of datasets in contrast to querying all datasets. Modified the corresponding unit test.

0.11.0
~~~~~~
* ADD #753: Allows uploading custom flows to OpenML via OpenML-Python.
* ADD #777: Allows running a flow on pandas dataframes (in addition to numpy arrays).
* ADD #888: Allow passing a `task_id` to `run_model_on_task`.
* ADD #894: Support caching of datasets using feather format as an option.
* ADD #929: Add ``edit_dataset`` and ``fork_dataset`` to allow editing and forking of uploaded datasets.
* ADD #866, #943: Add support for scikit-learn's `passthrough` and `drop` when uploading flows to
  OpenML.
* ADD #879: Add support for scikit-learn's MLP hyperparameter `layer_sizes`.
* ADD #894: Support caching of datasets using feather format as an option.
* ADD #945: PEP 561 compliance for distributing Type information.
* DOC #660: Remove nonexistent argument from docstring.
* DOC #901: The API reference now documents the config file and its options.
* DOC #912: API reference now shows `create_task`.
* DOC #954: Remove TODO text from documentation.
* DOC #960: document how to upload multiple ignore attributes.
* FIX #873: Fixes an issue which resulted in incorrect URLs when printing OpenML objects after
  switching the server.
* FIX #885: Logger no longer registered by default. Added utility functions to easily register
  logging to console and file.
* FIX #890: Correct the scaling of data in the SVM example.
* MAINT #371: ``list_evaluations`` default ``size`` changed from ``None`` to ``10_000``.
* MAINT #767: Source distribution installation is now unit-tested.
* MAINT #781: Add pre-commit and automated code formatting with black.
* MAINT #804: Rename arguments of list_evaluations to indicate they expect lists of ids.
* MAINT #836: OpenML supports only pandas version 1.0.0 or above.
* MAINT #865: OpenML no longer bundles test files in the source distribution.
* MAINT #881: Improve the error message for too-long URIs.
* MAINT #897: Dropping support for Python 3.5.
* MAINT #916: Adding support for Python 3.8.
* MAINT #920: Improve error messages for dataset upload.
* MAINT #921: Improve hangling of the OpenML server URL in the config file.
* MAINT #925: Improve error handling and error message when loading datasets.
* MAINT #928: Restructures the contributing documentation.
* MAINT #936: Adding support for scikit-learn 0.23.X.
* MAINT #945: Make OpenML-Python PEP562 compliant.
* MAINT #951: Converts TaskType class to a TaskType enum.

0.10.2
~~~~~~
* ADD #857: Adds task type ID to list_runs
* DOC #862: Added license BSD 3-Clause to each of the source files.

0.10.1
~~~~~~
* ADD #175: Automatically adds the docstring of scikit-learn objects to flow and its parameters.
* ADD #737: New evaluation listing call that includes the hyperparameter settings.
* ADD #744: It is now possible to only issue a warning and not raise an exception if the package
  versions for a flow are not met when deserializing it.
* ADD #783: The URL to download the predictions for a run is now stored in the run object.
* ADD #790: Adds the uploader name and id as new filtering options for ``list_evaluations``.
* ADD #792: New convenience function ``openml.flow.get_flow_id``.
* ADD #861: Debug-level log information now being written to a file in the cache directory (at most 2 MB).
* DOC #778: Introduces instructions on how to publish an extension to support other libraries
  than scikit-learn.
* DOC #785: The examples section is completely restructured into simple simple examples, advanced
  examples and examples showcasing the use of OpenML-Python to reproduce papers which were done
  with OpenML-Python.
* DOC #788: New example on manually iterating through the split of a task.
* DOC #789: Improve the usage of dataframes in the examples.
* DOC #791: New example for the paper *Efficient and Robust Automated Machine Learning* by Feurer
  et al. (2015).
* DOC #803: New example for the paper *Don’t  Rule  Out  Simple  Models Prematurely:
  A Large Scale  Benchmark Comparing Linear and Non-linear Classifiers in OpenML* by Benjamin
  Strang et al. (2018).
* DOC #808: New example demonstrating basic use cases of a dataset.
* DOC #810: New example demonstrating the use of benchmarking studies and suites.
* DOC #832: New example for the paper *Scalable Hyperparameter Transfer Learning* by
  Valerio Perrone et al. (2019)
* DOC #834: New example showing how to plot the loss surface for a support vector machine.
* FIX #305: Do not require the external version in the flow XML when loading an object.
* FIX #734: Better handling of *"old"* flows.
* FIX #736: Attach a StreamHandler to the openml logger instead of the root logger.
* FIX #758: Fixes an error which made the client API crash when loading a sparse data with
  categorical variables.
* FIX #779: Do not fail on corrupt pickle
* FIX #782: Assign the study id to the correct class attribute.
* FIX #819: Automatically convert column names to type string when uploading a dataset.
* FIX #820: Make ``__repr__`` work for datasets which do not have an id.
* MAINT #796: Rename an argument to make the function ``list_evaluations`` more consistent.
* MAINT #811: Print the full error message given by the server.
* MAINT #828: Create base class for OpenML entity classes.
* MAINT #829: Reduce the number of data conversion warnings.
* MAINT #831: Warn if there's an empty flow description when publishing a flow.
* MAINT #837: Also print the flow XML if a flow fails to validate.
* FIX #838: Fix list_evaluations_setups to work when evaluations are not a 100 multiple.
* FIX #847: Fixes an issue where the client API would crash when trying to download a dataset
  when there are no qualities available on the server.
* MAINT #849: Move logic of most different ``publish`` functions into the base class.
* MAINt #850: Remove outdated test code.

0.10.0
~~~~~~

* ADD #737: Add list_evaluations_setups to return hyperparameters along with list of evaluations.
* FIX #261: Test server is cleared of all files uploaded during unit testing.
* FIX #447: All files created by unit tests no longer persist in local.
* FIX #608: Fixing dataset_id referenced before assignment error in get_run function.
* FIX #447: All files created by unit tests are deleted after the completion of all unit tests.
* FIX #589: Fixing a bug that did not successfully upload the columns to ignore when creating and publishing a dataset.
* FIX #608: Fixing dataset_id referenced before assignment error in get_run function.
* DOC #639: More descriptive documention for function to convert array format.
* DOC #719: Add documentation on uploading tasks.
* ADD #687: Adds a function to retrieve the list of evaluation measures available.
* ADD #695: A function to retrieve all the data quality measures available.
* ADD #412: Add a function to trim flow names for scikit-learn flows.
* ADD #715: `list_evaluations` now has an option to sort evaluations by score (value).
* ADD #722: Automatic reinstantiation of flow in `run_model_on_task`. Clearer errors if that's not possible.
* ADD #412: The scikit-learn extension populates the short name field for flows.
* MAINT #726: Update examples to remove deprecation warnings from scikit-learn
* MAINT #752: Update OpenML-Python to be compatible with sklearn 0.21
* ADD #790: Add user ID and name to list_evaluations


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
* ADD #75, #653: Adds a pretty print for objects of the top-level classes.
* FIX #642: `check_datasets_active` now correctly also returns active status of deactivated datasets.
* FIX #304, #636: Allow serialization of numpy datatypes and list of lists of more types (e.g. bools, ints) for flows.
* FIX #651: Fixed a bug that would prevent openml-python from finding the user's config file.
* FIX #693: OpenML-Python uses liac-arff instead of scipy.io for loading task splits now.
* DOC #678: Better color scheme for code examples in documentation.
* DOC #681: Small improvements and removing list of missing functions.
* DOC #684: Add notice to examples that connect to the test server.
* DOC #688: Add new example on retrieving evaluations.
* DOC #691: Update contributing guidelines to use Github draft feature instead of tags in title.
* DOC #692: All functions are documented now.
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
