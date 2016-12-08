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

When downloading datasets, tasks, runs and flows, they will be cached to
retrieve them without calling the server later. As with the API key, the cache
directory can be either specified through the API or through the config file:

API:

.. code:: python

    >>> openml.config.set_cache_directory('~/.openml/cache')

Config file:

.. code:: bash

    cachedir = '~/.openml/cache'

~~~~~~~~~~~~~~~~~~~~~
Working with datasets
~~~~~~~~~~~~~~~~~~~~~

# TODO mention third, searching for tags
Datasets are a key concept in OpenML (see `OpenML documentation <openml.org/guide>`_).
Datasets are identified by IDs and can be accessed in two different ways:

1. In a list providing basic information on all datasets available on OpenML.
   This function will not download the actual dataset, but will instead download
   meta data which can be used to filter the datasets and retrieve a set of IDs.
2. A single dataset by its ID. A single dataset contains all meta information and the actual
   data in form of an .arff file. The .arff file will be converted into a numpy
   array by the OpenML Python API.

Listing datasets
~~~~~~~~~~~~~~~~

A common task when using OpenML is to find a set of datasets which fulfill
several criteria. They should for example have between 1,000 and 10,000
data points and at least five features.

.. code:: python

    >>> datasets = openml.datasets.list_datasets()

:meth:`openml.datasets.list_datasets` returns a list of dictionaries, we will
convert it into a `pandas dataframe <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
to have better visualization:

.. code:: python

    >>> import pandas as pd
    >>> datasets = pd.DataFrame(datasets)
    >>> datasets.set_index('did', inplace=True)

We have access to the following properties of the datasets:

    >>> print(datasets.columns)
    Index([                 u'MajorityClassSize',
                  u'MaxNominalAttDistinctValues',
                            u'MinorityClassSize',
                                u'NumBinaryAtts',
                              u'NumberOfClasses',
                             u'NumberOfFeatures',
                            u'NumberOfInstances',
           u'NumberOfInstancesWithMissingValues',
                        u'NumberOfMissingValues',
                      u'NumberOfNumericFeatures',
                     u'NumberOfSymbolicFeatures',
                                       u'format',
                                         u'name',
                                       u'status'],
          dtype='object')

and can see the first data point:

    >>> print(datasets.iloc[0])
    MajorityClassSize                        684
    MaxNominalAttDistinctValues               10
    MinorityClassSize                          0
    NumBinaryAtts                             14
    NumberOfClasses                            6
    NumberOfFeatures                          39
    NumberOfInstances                        898
    NumberOfInstancesWithMissingValues         0
    NumberOfMissingValues                      0
    NumberOfNumericFeatures                    6
    NumberOfSymbolicFeatures                  32
    format                                  ARFF
    name                                  anneal
    status                                active
    Name: 1, dtype: object

We can now filter the data:

    >>> filter = (datasets.NumberOfInstances > 1000) & (datasets.NumberOfFeatures > 5)
    >>> filtered_datasets = datasets.loc[filter]
    >>> dataset_indices = list(filtered_datasets.index)
    >>> print(dataset_indices)                                                  # doctest: +SKIP
    [3, 6, 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 30, 32, 36, 38, 44,
    ... 5291, 5293, 5295, 5296, 5297, 5301, 5587, 5648, 5889]

and get a list of dataset indices which can be used in a next step.

Downloading datasets
~~~~~~~~~~~~~~~~~~~~

We can now use the dataset IDs to download all datasets by their IDs. Let's
first look at how to download a single dataset and what can be done with the
dataset object:

.. code:: python

    >>> dataset_id = 23
    >>> dataset = openml.datasets.get_dataset(dataset_id)

Properties of the dataset are stored as member variables:

.. code:: python

    >>> print(dataset.__dict__)                                               # doctest: +SKIP
    {'upload_date': u'2014-04-06 23:21:03', 'md5_cheksum': u'3149646ecff276abac3e892d1556655f', 'creator': None, 'citation': None, 'tag': [u'study_1', u'study_7', u'uci'], 'version_label': u'1', 'contributor': None, 'paper_url': None, 'original_data_url': None, 'id': 23, 'collection_date': None, 'row_id_attribute': None, 'version': 1, 'data_pickle_file': '/home/matthias/.openml/cache/datasets/23/dataset.pkl', 'default_target_attribute': u'Contraceptive_method_used', 'description': u"**Author**:   \n**Source**: Unknown -   \n**Please cite**:   \n\n1. Title: Contraceptive Method Choice\n \n 2. Sources:\n    (a) Origin:  This dataset is a subset of the 1987 National Indonesia\n                 Contraceptive Prevalence Survey\n    (b) Creator: Tjen-Sien Lim (limt@stat.wisc.edu)\n    (c) Donor:   Tjen-Sien Lim (limt@stat.wisc.edu)\n    (c) Date:    June 7, 1997\n \n 3. Past Usage:\n    Lim, T.-S., Loh, W.-Y. & Shih, Y.-S. (1999). A Comparison of\n    Prediction Accuracy, Complexity, and Training Time of Thirty-three\n    Old and New Classification Algorithms. Machine Learning. Forthcoming.\n    (ftp://ftp.stat.wisc.edu/pub/loh/treeprogs/quest1.7/mach1317.pdf or\n    (http://www.stat.wisc.edu/~limt/mach1317.pdf)\n \n 4. Relevant Information:\n    This dataset is a subset of the 1987 National Indonesia Contraceptive\n    Prevalence Survey. The samples are married women who were either not \n    pregnant or do not know if they were at the time of interview. The \n    problem is to predict the current contraceptive method choice \n    (no use, long-term methods, or short-term methods) of a woman based \n    on her demographic and socio-economic characteristics.\n \n 5. Number of Instances: 1473\n \n 6. Number of Attributes: 10 (including the class attribute)\n \n 7. Attribute Information:\n \n    1. Wife's age                     (numerical)\n    2. Wife's education               (categorical)      1=low, 2, 3, 4=high\n    3. Husband's education            (categorical)      1=low, 2, 3, 4=high\n    4. Number of children ever born   (numerical)\n    5. Wife's religion                (binary)           0=Non-Islam, 1=Islam\n    6. Wife's now working?            (binary)           0=Yes, 1=No\n    7. Husband's occupation           (categorical)      1, 2, 3, 4\n    8. Standard-of-living index       (categorical)      1=low, 2, 3, 4=high\n    9. Media exposure                 (binary)           0=Good, 1=Not good\n    10. Contraceptive method used     (class attribute)  1=No-use \n                                                         2=Long-term\n                                                         3=Short-term\n \n 8. Missing Attribute Values: None\n\n Information about the dataset\n CLASSTYPE: nominal\n CLASSINDEX: last", 'format': u'ARFF', 'visibility': u'public', 'update_comment': None, 'licence': u'Public', 'name': u'cmc', 'language': None, 'url': u'http://www.openml.org/data/download/23/dataset_23_cmc.arff', 'data_file': '~/.openml/cache/datasets/23/dataset.arff', 'ignore_attributes': None}

Next, to obtain the data matrix:

.. code:: python

    >>> X = dataset.get_data()
    >>> print(X.shape, X.dtype)
    ((1473, 10), dtype('float32'))

which returns the dataset as a np.ndarray with dtype :python:`np.float32`.
In case the data is sparse, a scipy.sparse.csr matrix is returned. All nominal
variables are encoded as integers, the inverse encoding can be retrieved via:

.. code:: python

    >>> X, names = dataset.get_data(return_attribute_names=True)
    >>> print(names)
    [u'Wifes_age', u'Wifes_education', u'Husbands_education', u'Number_of_children_ever_born', u'Wifes_religion', u'Wifes_now_working%3F', u'Husbands_occupation', u'Standard-of-living_index', u'Media_exposure', u'Contraceptive_method_used']

Most times, having a single data matrix :python:`X` is not enough. Two 
useful arguments are :python:`target` and
:python:`return_categorical_indicator`. :python:`target` makes
:meth:`get_data()` return :python:`X` and :python:`y`
seperate; :python:`return_categorical_indicator` makes
:meth:`get_data()` return a boolean array which indicate
which attributes are categorical (and should be one hot encoded if necessary.)

.. code:: python

    >>> X, y, categorical = dataset.get_data(
    ... target=dataset.default_target_attribute,
    ... return_categorical_indicator=True)
    >>> print(X.shape, y.shape)
    ((1473, 9), (1473,))
    >>> print(categorical)
    [False, True, True, False, True, True, True, True, True]

In case you are working with `scikit-learn
<http://scikit-learn>`_, you can use this data right away:

.. code:: python

    >>> from sklearn import preprocessing, ensemble
    >>> enc = preprocessing.OneHotEncoder(categorical_features=categorical)
    >>> print(enc)
    OneHotEncoder(categorical_features=[False, True, True, False, True, True, True, True, True],
           dtype=<type 'float'>, handle_unknown='error', n_values='auto',
           sparse=True)
    >>> X = enc.fit_transform(X).todense()
    >>> clf = ensemble.RandomForestClassifier()
    >>> clf.fit(X, y)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)

When you have to retrieve several datasets, you can use the convenience function
:meth:`openml.datasets.get_datasets()`, which downloads all datasets given by
a list of IDs:

    >>> ids = [12, 14, 16, 18, 20, 22]
    >>> datasets = openml.datasets.get_datasets(ids)
    >>> print(datasets[0].name)
    mfeat-factors

~~~~~~~~~~~~~~~~~~
Working with tasks
~~~~~~~~~~~~~~~~~~

#TODO put a link to the OpenML documentation here! Link the Task functions and
the task class

While datasets provide the most basic information for a machine learning task,
they do not provide enough information for a reproducible machine learning
experiment. A task defines how to split the dataset into a train and test set,
whether to use several disjoint train and test splits (cross-validation) and
whether this should be repeated several times. Also, the task defines a target
metric for which a flow should be optimized.

Just like datasets, tasks are identified by IDs and can be accessed in three
different ways:

1. In a list providing basic information on all tasks available on OpenML.
   This function will not download the actual tasks, but will instead download
   meta data that can be used to filter the tasks and retrieve a set of IDs.
2. By functions only list a subset of all available tasks, restricted either by
   their :TODO:`task_type`, :TODO:`tag` or :TODO:`check_for_more`.
3. A single task by its ID. It contains all meta information, the target metric,
   the splits and an iterator which can be used to access the splits in a
   useful manner.

You can also read more about tasks in the `OpenML guide <http://www.openml.org/guide>`_.

Listing tasks
~~~~~~~~~~~~~

Once we decide on the datasets we want to work on, we have to download the
corresponding tasks. Tasks can be pre-filtered by either by a task type or
a tag.

So far, this package only supports supervised classification tasks (task
type :python:`1`) and supervised regression tasks (task type :python:`2`) #TODO check this
We desribe how to find other task types in the subsection `Finding out task types`_
and are happy to receive contributions that help us to support all other task
types.

The most natural way to retrieve tasks is by their task type. In this example
we will use the most commonly studied machine learning supervised
classification task (task type :python:`1`):

.. code:: python

    >>> tasks = openml.tasks.list_tasks_by_type(1)

Let's find out more about the datasets:

.. code:: python

    >>> import pandas as pd
    >>> tasks = pd.DataFrame(tasks)
    >>> tasks.set_index('tid', inplace=True)
    >>> print(tasks.columns)
    Index([         u'cost_matrix',                  u'did',
           u'estimation_procedure',  u'evaluation_measures',
                           u'name',          u'source_data',
                         u'status',       u'target_feature',
                      u'task_type'],
          dtype='object')

Now we can restrict the tasks to all tasks with the desired resampling strategy:

# TODO add something about the different resampling strategies implemented!

.. code:: python

    >>> filter = tasks.estimation_procedure == '10-fold Crossvalidation'
    >>> filtered_tasks = tasks[filter]
    >>> filtered_tasks = list(filtered_tasks.index)
    >>> print(filtered_tasks)                               # doctest: +SKIP
    [1, 2, 3, 4, 5, 6, 7, 8, 9, ... 10105, 10106, 10107, 10109, 10111, 13907, 13918]

Resampling strategies can be found on the `OpenML Website <http://www.openml.org/search?type=measure&q=estimation%20procedure>`_
or programatically as described in `Finding out evaluation strategies and target metrics`_.

Finally, we can check whether there is a task for each dataset that we want to
use in our study. If this is not the case, tasks can be created on the
`OpenML website <openml.org/tasks/create>`_.

The rest of this subsection deals with accessing a list of tasks by tags and
without any restriction.

A list of tasks, filtered tags, can be retrieved via:

.. code:: python

    >>> tasks = openml.tasks.list_tasks_by_tag('study_1')

:meth:`openml.tasks.list_tasks_by_tag` returns a list of dictionaries, we will
convert it into a `pandas dataframe <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
to have better visualization:

.. code:: python

    >>> import pandas as pd
    >>> tasks = pd.DataFrame(tasks)
    >>> tasks.set_index('tid', inplace=True)

As before, we have to check whether there is a task for each dataset that we
want to work with. In addition, we have to make sure to use only tasks with the
desired task type:

#TODO this doesn't look nice, we should have a constant for each known task,
dynamically created by the task type available (but when do we know that we
can savely use the api connector? what to do if we do not have an internet
connection? Maybe have this statically in the program and check from time to
time if there is something new (via a unit test?)?, the same holds true for
the resampling strategies available!)

.. code:: python

    >>> filter = tasks.task_type == 'Supervised Classification'
    >>> filtered_tasks = tasks[filter]
    >>> print(len(filtered_tasks))                                  # doctest: +SKIP
    2599

Finally, it is also possible to list all tasks on OpenML with:

.. code:: python

    >>> tasks = openml.tasks.list_tasks()
    >>> print(len(tasks))                       # doctest: +SKIP
    29757

Downloading tasks
~~~~~~~~~~~~~~~~~

Downloading tasks works similar to downloading datasets. We provide two
functions for this, one which downloads only a single task by its ID,
and one which takes a list of IDs and downloads all of these tasks:

.. code:: python

    >>> task_id = 1
    >>> task = openml.tasks.get_task(task_id)

Properties of the task are stored as member variables:

.. code:: python

    >>> print(task.__dict__)
    {'target_feature': u'class', 'task_type': u'Supervised Classification', 'task_id': 1, 'estimation_procedure': {'type': u'crossvalidation', 'data_splits_url': u'http://www.openml.org/api_splits/get/1/Task_1_splits.arff', 'parameters': {u'number_repeats': u'1', u'percentage': '', u'stratified_sampling': u'true', u'number_folds': u'10'}}, 'class_labels': [u'1', u'2', u'3', u'4', u'5', u'U'], 'cost_matrix': None, 'evaluation_measure': u'predictive_accuracy', 'dataset_id': 1, 'estimation_parameters': {u'number_repeats': u'1', u'percentage': '', u'stratified_sampling': u'true', u'number_folds': u'10'}}

And with a list of task IDs:

.. code:: python

    >>> ids = [12, 14, 16, 18, 20, 22]
    >>> tasks = openml.tasks.get_tasks(ids)
    >>> print(tasks[0])

~~~~~~~~~~~~~~~~~~~~~~~
Finding out tasks types
~~~~~~~~~~~~~~~~~~~~~~~

Not yet supported by the API. Please use the OpenML website.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Finding out evaluation strategies and target metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not yet supported by the API. Please use the OpenML website.

~~~~~~~~~~~~~~~
Using the cache
~~~~~~~~~~~~~~~

Downloading all datasets, tasks and split every time a get function is called
would prohibit a user to interact with the API in an exploratory manner.
OpenML is designed in a way that certain entities are immutable once created.
This allows the python package to cache datasets, tasks, splits and runs locally
for fast retrieval. Another benefit is that the API can be used normally on a
compute cluster without internet access (:ref:`see below`).

Currently, the following objects are cached:

* datasets
    * dataset arff. In order to reduce parsing time, the data is serialized to
      disk in a binary format (using the `pickle library <https://docs.python.org/2/library/pickle.html>`_.
    * dataset descriptions
    * more?
* tasks
    * task description
    * split arff. TODO are they cached?
* runs
    * run description

Run predictions are not cached yet. Flow ojects cannot yet be downloaded and are
therefore not cached.

Configuring the cache
~~~~~~~~~~~~~~~~~~~~~

Configuring the cache works as described in the subsection `Connecting to the OpenML server`_:
It can be done either through the API:

.. code:: python

    >>> openml.config.set_cache_directory('~/.openml/cache')

or the config file:

.. code:: bash

    cachedir = '~/.openml/cache'


Clearing the cache
~~~~~~~~~~~~~~~~~~

Currently, there is no programmatic way to interact with the cache and we do not
plan to implement one. If you have any use case for this, please open an issue
on the `issue tracker <https://github.com/openml/openml-python/issues>`_.

# TODO check that the cache is in a consistent state!
In case the cache gets too large, you can manually delete unnecessary files.
Make sure that you always delete a complete entity, for example the whole
directory caching a dataset named after the datasets ID.

~~~~~~~~~~~~~~~~~~~~~~~~~~~
Working with Flows and Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks and datasets allow us to download all information to run an experiment
locally. In order to upload and share results of such an experiment we need
the concepts of flows and runs.

Flows are descriptions of something runable which does the machine learning.
A flow contains all information to set up the necessary machine learning
library and its dependencies as well as all possible parameters.

A run is the outcome of running a flow on a task. It contains all parameter
settings for the flow, a setup string (most likely a command line call) and all
predictions of that run. When a run is uploaded to the server, the server
automatically calculates several metrics which can be used to compare the
performance of different flows to each other.

Creating a flow
~~~~~~~~~~~~~~~

So far, a flow in the OpenML python API is expected to be an estimator object
following the `scikit-learn estimator API <http://scikit-learn.org/dev/developers/contributing.html#apis-of-scikit-learn-objects>`_.
Thus, it needs to implement a :python:`fit()`, :python:`predict()`,
:python:`set_params()` and :python:`get_params()` method.

The only mandatory argument to construct a flow is an estimator object. It is of
course possible and also useful to pass other arguments to the constructor, for
example a description or the name of the creator:

.. code:: python

    >>> from openml import OpenMLFlow
    >>> model = ensemble.RandomForestClassifier()
    >>> flow = OpenMLFlow(model, description='Out-of-the-box scikit-learn '
    ...                                      'random forest classifier',
    ...                   creator='Matthias Feurer')
    >>> print(flow)
    {'description': 'Out-of-the-box scikit-learn random forest classifier', 'creator': 'Matthias Feurer', 'external_version': 'sklearn_0.16.1', 'source': 'FIXME DEFINE PYTHON FLOW', 'tag': None, 'upoader': None, 'model': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False), 'id': None, 'name': 'sklearn.ensemble.forest.RandomForestClassifier'}

Prior to using a flow in experiments, it has to be pushed to the OpenML
website. The python OpenML API uses methods named :python:`publish()` for this.
It takes no arguments, but uses information from the estimator object to
obtain all necessary information. The information is pushed to OpenML, and the
flow gets assigned a flow ID.

.. code:: python

    >>> flow.publish()
    # What happens here? What should it return?

Running a flow on a task
~~~~~~~~~~~~~~~~~~~~~~~~

We can now use the created flow fo finally run a machine learning model on a
task and upload the results to the OpenML website. For that we use the function
:meth:`openml.runs.run_task`. It only accepts two arguments and does what the
name suggests.

.. code:: python

    >>> task_id = 12
    >>> run = openml.runs.run_task(task_id, model)
    >>> print(run)

As for flows, the run must be published so that it can be used by others on
OpenML:

.. code:: python

    >>> run.publish()
    # What happens here? What should it return?

Retrieving results from OpenML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~









