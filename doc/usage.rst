:orphan:

.. _usage:

.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

***********
Basic Usage
***********

This document will guide you through the most important functions and classes
in the OpenML Python API. Throughout this document, we will use
`pandas <http://pandas.pydata.org/>`_ to format and filter tables.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Connecting to the OpenML server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OpenML server can only be accessed by users who have signed up on the OpenML
platform. If you don't have an account yet,
`sign up now <http://openml.org/register>`_. You will receive an API key, which
will authenticate you to the server and allow you to download and upload
datasets, tasks, runs and flows. There are two ways of providing the API key
to the OpenML API package. The first option is to specify the API key
programmatically after loading the package:

.. code:: python

    >>> import openml
    >>> apikey = 'Your API key'
    >>> openml.config.apikey = apikey

The second option is to create a config file:

.. code:: bash

    apikey = qxlfpbeaudtprb23985hcqlfoebairtd

The config file must be in the directory :bash:`~/.openml/config` and 
exist prior to importing the openml module.

..
    >>> openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'

When downloading datasets, tasks, runs and flows, they will be cached to
retrieve them without calling the server later. As with the API key, the cache
directory can be either specified through the API or through the config file:

API:

.. code:: python

    >>> import os
    >>> openml.config.set_cache_directory(os.path.expanduser('~/.openml/cache'))

Config file:

.. code:: bash

    cachedir = '~/.openml/cache'


~~~~~~~~~~~~
Key concepts
~~~~~~~~~~~~

OpenML contains several key concepts which it needs to make machine learning
research shareable. A machine learning experiment consists of one or several
**runs**, which describe the performance of an algorithm (called a **flow** in
OpenML), its hyperparameter settings (called a **setup**) on a **task**. A
**Task** is the combination of a **dataset**, a split and an evaluation
metric. In this user guide we will go through listing and exploring existing
**tasks** to actually running machine learning algorithms on them. In a further
user guide we will examine how to search through **datasets** in order to curate
a list of **tasks**.

~~~~~~~~~~~~~~~~~~
Working with tasks
~~~~~~~~~~~~~~~~~~

You can think of a task as an experimentation protocol, describing how to apply
a machine learning model to a dataset in a way that it is comparable with the
results of others (more on how to do that further down).Tasks are containers,
defining which dataset to use, what kind of task we're solving (regression,
classification, clustering, etc...) and which column to predict. Furthermore,
it also describes how to split the dataset into a train and test set, whether
to use several disjoint train and test splits (cross-validation) and whether
this should be repeated several times. Also, the task defines a target metric
for which a flow should be optimized.

Tasks are identified by IDs and can be accessed in two different ways:

1. In a list providing basic information on all tasks available on OpenML.
   This function will not download the actual tasks, but will instead download
   meta data that can be used to filter the tasks and retrieve a set of IDs.
   We can filter this list, for example, we can only list tasks having a special
   tag or only tasks for a specific target such as *supervised classification*.

2. A single task by its ID. It contains all meta information, the target metric,
   the splits and an iterator which can be used to access the splits in a
   useful manner.

You can also read more about tasks in the `OpenML guide <http://www.openml.org/guide>`_.

Listing tasks
~~~~~~~~~~~~~

So far, this package only supports *supervised classification* tasks (task
type :python:`1`). Therefore, well will start by simply listing all these tasks:

.. code:: python

    >>> tasks = openml.tasks.list_tasks(task_type_id=1)

:meth:`openml.tasks.list_tasks` returns a dictionary of dictionaries, we convert
it into a
`pandas dataframe <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
to have better visualization and easier access:

.. code:: python

    >>> import pandas as pd
    >>> tasks = pd.DataFrame.from_dict(tasks, orient='index')
    >>> print(tasks.columns)
    Index(['tid', 'ttid', 'did', 'name', 'task_type', 'status',
           'estimation_procedure', 'evaluation_measures', 'source_data',
           'target_feature', 'MajorityClassSize', 'MaxNominalAttDistinctValues',
           'MinorityClassSize', 'NumberOfClasses', 'NumberOfFeatures',
           'NumberOfInstances', 'NumberOfInstancesWithMissingValues',
           'NumberOfMissingValues', 'NumberOfNumericFeatures',
           'NumberOfSymbolicFeatures', 'cost_matrix'],
          dtype='object')

We can filter the list of tasks to only contain datasets with more than
500 samples, but less than 1000 samples:

.. code:: python

    >>> filtered_tasks = tasks.query('NumberOfInstances > 500 and NumberOfInstances < 1000')
    >>> print(list(filtered_tasks.index))                               # doctest: +SKIP
    [2, 11, 15, 29, 37, 41, 49, 53, ..., 146597, 146600, 146605]
    >>> print(len(filtered_tasks))
    210

Then, we can further restrict the tasks to all have the same resampling
strategy:

.. code:: python

    >>> filtered_tasks = filtered_tasks.query('estimation_procedure == "10-fold Crossvalidation"')
    >>> print(list(filtered_tasks.index))                               # doctest: +SKIP
    [2, 11, 15, 29, 37, 41, 49, 53, ..., 146231, 146238, 146241]
    >>> print(len(filtered_tasks))                                      # doctest: +SKIP
    107

Resampling strategies can be found on the `OpenML Website <http://www.openml.org/search?type=measure&q=estimation%20procedure>`_.

Similar to listing tasks by task type, we can list tasks by tags:

.. code:: python

    >>> tasks = openml.tasks.list_tasks(tag='OpenML100')
    >>> tasks = pd.DataFrame.from_dict(tasks, orient='index')

*OpenML 100* is a curated list of 100 tasks to start using OpenML. They are all
supervised classification tasks with more than 500 instances and less than 50000
instances per task. To make things easier, the tasks do not contain highly
unbalanced data and sparse data. However, the tasks include missing values and
categorical features. You can find out more about the *OpenML 100* on
`the OpenML benchmarking page <https://www.openml.org/guide/benchmark>`_.

Finally, it is also possible to list all tasks on OpenML with:

.. code:: python

    >>> tasks = openml.tasks.list_tasks()
    >>> print(len(tasks))                       # doctest: +SKIP
    46067

Downloading tasks
~~~~~~~~~~~~~~~~~

We provide two functions to download tasks, one which downloads only a single
task by its ID, and one which takes a list of IDs and downloads all of these
tasks:

.. code:: python

    >>> task_id = 2
    >>> task = openml.tasks.get_task(task_id)

Properties of the task are stored as member variables:

.. code:: python

    >>> from pprint import pprint
    >>> pprint(vars(task))
    {'class_labels': ['1', '2', '3', '4', '5', 'U'],
     'cost_matrix': None,
     'dataset_id': 2,
     'estimation_parameters': {'number_folds': '10',
                               'number_repeats': '1',
                               'percentage': '',
                               'stratified_sampling': 'true'},
     'estimation_procedure': {'data_splits_url': 'https://www.openml.org/api_splits/get/2/Task_2_splits.arff',
                              'parameters': {'number_folds': '10',
                                             'number_repeats': '1',
                                             'percentage': '',
                                             'stratified_sampling': 'true'},
                              'type': 'crossvalidation'},
     'evaluation_measure': 'predictive_accuracy',
     'split': None,
     'target_name': 'class',
     'task_id': 2,
     'task_type': 'Supervised Classification',
     'task_type_id': 1}

And:

.. code:: python

    >>> ids = [2, 11, 15, 29, 37, 41, 49, 53]
    >>> tasks = openml.tasks.get_tasks(ids)
    >>> pprint(tasks[0])                           # doctest: +SKIP

~~~~~~~~~~~~~
Creating runs
~~~~~~~~~~~~~

In order to upload and share results of running a machine learning algorithm
on a task, we need to create an :class:`~openml.OpenMLRun`. A run object can
be created by running a :class:`~openml.OpenMLFlow` or a scikit-learn compatible
model on a task. We will focus on the simpler example of running a
scikit-learn model.

Flows are descriptions of something runable which does the machine learning.
A flow contains all information to set up the necessary machine learning
library and its dependencies as well as all possible parameters.

A run is the outcome of running a flow on a task. It contains all parameter
settings for the flow, a setup string (most likely a command line call) and all
predictions of that run. When a run is uploaded to the server, the server
automatically calculates several metrics which can be used to compare the
performance of different flows to each other.

So far, the OpenML python connector works only with estimator objects following
the `scikit-learn estimator API <http://scikit-learn.org/dev/developers/contributing.html#apis-of-scikit-learn-objects>`_.
Those can be directly run on a task, and a flow will automatically be created or
downloaded from the server if it already exists.

Running a model
~~~~~~~~~~~~~~~

.. code:: python

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> task = openml.tasks.get_task(12)
    >>> run = openml.runs.run_model_on_task(task, model)
    >>> pprint(vars(run), depth=2)                             # doctest: +SKIP
    {'data_content': [...],
     'dataset_id': 12,
     'error_message': None,
     'evaluations': None,
     'flow': None,
     'flow_id': 7257,
     'flow_name': None,
     'fold_evaluations': defaultdict(<function _run_task_get_arffcontent.<locals>.<lambda> at 0x7fb88981b9d8>,
                                     {'predictive_accuracy': defaultdict(<class 'dict'>,
                                                                         {0: {0: 0.94499999999999995,
                                                                              1: 0.94499999999999995,
                                                                              2: 0.94499999999999995,
                                                                              3: 0.96499999999999997,
                                                                              4: 0.92500000000000004,
                                                                              5: 0.96499999999999997,
                                                                              6: 0.94999999999999996,
                                                                              7: 0.96999999999999997,
                                                                              8: 0.93999999999999995,
                                                                              9: 0.95499999999999996}}),
                                      'usercpu_time_millis': defaultdict(<class 'dict'>,
                                                                         {0: {0: 110.4880920000042,
                                                                              1: 105.7469440000034,
                                                                              2: 107.4153629999941,
                                                                              3: 105.1104170000059,
                                                                              4: 104.02388900000403,
                                                                              5: 105.17172800000196,
                                                                              6: 109.00792000001047,
                                                                              7: 107.49670599999206,
                                                                              8: 107.34138000000115,
                                                                              9: 104.78881499999915}}),
                                      'usercpu_time_millis_testing': defaultdict(<class 'dict'>,
                                                                                 {0: {0: 3.6470320000034917,
                                                                                      1: 3.5307810000020368,
                                                                                      2: 3.5432540000002177,
                                                                                      3: 3.5460690000022055,
                                                                                      4: 3.5634600000022942,
                                                                                      5: 3.906016000001955,
                                                                                      6: 3.6680000000046675,
                                                                                      7: 3.643865999997331,
                                                                                      8: 3.4515420000005292,
                                                                                      9: 3.461469000001216}}),
                                      'usercpu_time_millis_training': defaultdict(<class 'dict'>,
                                                                                  {0: {0: 106.84106000000071,
                                                                                       1: 102.21616300000136,
                                                                                       2: 103.87210899999388,
                                                                                       3: 101.56434800000369,
                                                                                       4: 100.46042900000174,
                                                                                       5: 101.26571200000001,
                                                                                       6: 105.3399200000058,
                                                                                       7: 103.85283999999473,
                                                                                       8: 103.88983800000062,
                                                                                       9: 101.32734599999793}})}),
     'model': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=43934,
                verbose=0, warm_start=False),
     'output_files': None,
     'parameter_settings': [...],
     'predictions_url': None,
     'run_id': None,
     'sample_evaluations': None,
     'setup_id': None,
     'setup_string': None,
     'tags': [...],
     'task': None,
     'task_evaluation_measure': None,
     'task_id': 12,
     'task_type': None,
     'trace_attributes': None,
     'trace_content': None,
     'uploader': None,
     'uploader_name': None}

So far the run is only available locally. By calling the publish function, the
run is send to the OpenML server:

.. code:: python

    >>> run.publish()                                          # doctest: +SKIP
    <openml.runs.run.OpenMLRun at 0x7fb8953d72e8>

We can now also inspect the flow object which was automatically created:

.. code:: python

    >>> flow = openml.flows.get_flow(run.flow_id)
    >>> pprint(vars(flow), depth=1)                             # doctest: +SKIP
    {'binary_format': None,
     'binary_md5': None,
     'binary_url': None,
     'class_name': 'sklearn.ensemble.forest.RandomForestClassifier',
     'components': OrderedDict(),
     'custom_name': None,
     'dependencies': 'sklearn==0.18.2\nnumpy>=1.6.1\nscipy>=0.9',
     'description': 'Automatically created scikit-learn flow.',
     'external_version': 'openml==0.6.0,sklearn==0.18.2',
     'flow_id': 7257,
     'language': 'English',
     'model': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False),
     'name': 'sklearn.ensemble.forest.RandomForestClassifier',
     'parameters': OrderedDict([...]),
     'parameters_meta_info': OrderedDict([...]),
     'tags': [...],
     'upload_date': '2017-10-09T10:20:40',
     'uploader': '1159',
     'version': '29'}

Advanced topics
~~~~~~~~~~~~~~~

We are working on tutorials for the following topics:

* Querying datasets
* Uploading datasets
* Creating tasks
* Working offline
* Analyzing large amounts of results
