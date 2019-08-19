:orphan:

.. _usage:

.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

**********
User Guide
**********

This document will guide you through the most important use cases, functions
and classes in the OpenML Python API. Throughout this document, we will use
`pandas <http://pandas.pydata.org/>`_ to format and filter tables.

~~~~~~~~~~~~~~~~~~~~~~
Installation & Set up
~~~~~~~~~~~~~~~~~~~~~~

The OpenML Python package is a connector to `OpenML <https://www.openml.org/>`_.
It allows to use and share datasets and tasks, run
machine learning algorithms on them and then share the results online.

The following tutorial gives a short introduction on how to install and set up
the OpenML python connector, followed up by a simple example.

* `Introduction <examples/introduction_tutorial.html>`_


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

A further explanation is given in the
`OpenML user guide <https://openml.github.io/OpenML/#concepts>`_.

~~~~~~~~~~~~~~~~~~
Working with tasks
~~~~~~~~~~~~~~~~~~

You can think of a task as an experimentation protocol, describing how to apply
a machine learning model to a dataset in a way that it is comparable with the
results of others (more on how to do that further down). Tasks are containers,
defining which dataset to use, what kind of task we're solving (regression,
classification, clustering, etc...) and which column to predict. Furthermore,
it also describes how to split the dataset into a train and test set, whether
to use several disjoint train and test splits (cross-validation) and whether
this should be repeated several times. Also, the task defines a target metric
for which a flow should be optimized.

Below you can find our tutorial regarding tasks and if you want to know more
you can read the `OpenML guide <https://docs.openml.org/#tasks>`_:

* `Tasks <examples/tasks_tutorial.html>`_

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running machine learning algorithms and uploading results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

The next tutorial covers how to train different machine learning models,
how to run machine learning models on OpenML data and how to share the results:

* `Flows and Runs <examples/flows_and_runs_tutorial.html>`_

~~~~~~~~
Datasets
~~~~~~~~

OpenML provides a large collection of datasets and the benchmark
"`OpenML100 <https://docs.openml.org/benchmark/>`_" which consists of a curated
list of datasets.

You can find the dataset that best fits your requirements by making use of the
available metadata. The tutorial which follows explains how to get a list of
datasets, how to filter the list to find the dataset that suits your
requirements and how to download a dataset:

* `Filter and explore datasets <examples/datasets_tutorial.html>`_

OpenML is about sharing machine learning results and the datasets they were
obtained on. Learn how to share your datasets in the following tutorial:

* `Upload a dataset <examples/create_upload_tutorial.html>`_

~~~~~~~~~~~~~~~~~~~~~~~
Extending OpenML-Python
~~~~~~~~~~~~~~~~~~~~~~~

OpenML-Python provides an extension interface to connect other machine learning libraries than
scikit-learn to OpenML. Please check the :ref:`api_extensions` and use the
scikit-learn extension in :class:`openml.extensions.sklearn.SklearnExtension` as a starting point.

~~~~~~~~~~~~~~~
Advanced topics
~~~~~~~~~~~~~~~

We are working on tutorials for the following topics:

* Querying datasets (TODO)
* Creating tasks (TODO)
* Working offline (TODO)
* Analyzing large amounts of results (TODO)
