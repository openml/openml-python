# User Guide

This document will guide you through the most important use cases,
functions and classes in the OpenML Python API. Throughout this
document, we will use [pandas](https://pandas.pydata.org/) to format and
filter tables.

## Installation

The OpenML Python package is a connector to
[OpenML](https://www.openml.org/). It allows you to use and share
datasets and tasks, run machine learning algorithms on them and then
share the results online.

The ["intruduction tutorial and setup"][intro] tutorial gives a short introduction on how to install and
set up the OpenML Python connector, followed up by a simple example.

## Configuration

The configuration file resides in a directory `.config/openml` in the
home directory of the user and is called config (More specifically, it
resides in the [configuration directory specified by the XDGB Base
Directory
Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)).
It consists of `key = value` pairs which are separated by newlines. The
following keys are defined:

- apikey: required to access the server. The [introduction tutorial][intro] describes how to obtain an API key.
- server: the server to connect to (default: `http://www.openml.org`).
          For connection to the test server, set this to `test.openml.org`.
- cachedir: the root folder where the cache file directories should be created.
    If not given, will default to `~/.openml/cache`
- avoid_duplicate_runs: if set to `True` (default), when `run_flow_on_task` or similar methods
            are called a lookup is performed to see if there already
            exists such a run on the server. If so, download those
            results instead.
- retry_policy: Defines how to react when the server is unavailable or
            experiencing high load. It determines both how often to
            attempt to reconnect and how quickly to do so. Please don't
            use `human` in an automated script that you run more than
            one instance of, it might increase the time to complete your
            jobs and that of others. One of:
            -   human (default): For people running openml in interactive
                fashion. Try only a few times, but in quick succession.
            -   robot: For people using openml in an automated fashion. Keep
                trying to reconnect for a longer time, quickly increasing
                the time between retries.

- connection_n_retries: number of times to retry a request if they fail. 
Default depends on retry_policy (5 for `human`, 50 for `robot`)
- verbosity: the level of output:
      -   0: normal output
      -   1: info output
      -   2: debug output

This file is easily configurable by the `openml` command line interface.
To see where the file is stored, and what its values are, use openml
configure none. 

## Docker

It is also possible to try out the latest development version of
`openml-python` with docker:

``` bash
docker run -it openml/openml-python
```

See the [openml-python docker
documentation](https://github.com/openml/openml-python/blob/main/docker/readme.md)
for more information.

## Key concepts

OpenML contains several key concepts which it needs to make machine
learning research shareable. A machine learning experiment consists of
one or several **runs**, which describe the performance of an algorithm
(called a **flow** in OpenML), its hyperparameter settings (called a
**setup**) on a **task**. A **Task** is the combination of a
**dataset**, a split and an evaluation metric. In this user guide we
will go through listing and exploring existing **tasks** to actually
running machine learning algorithms on them. In a further user guide we
will examine how to search through **datasets** in order to curate a
list of **tasks**.

A further explanation is given in the [OpenML user
guide](https://openml.github.io/OpenML/#concepts).

## Working with tasks

You can think of a task as an experimentation protocol, describing how
to apply a machine learning model to a dataset in a way that is
comparable with the results of others (more on how to do that further
down). Tasks are containers, defining which dataset to use, what kind of
task we\'re solving (regression, classification, clustering, etc\...)
and which column to predict. Furthermore, it also describes how to split
the dataset into a train and test set, whether to use several disjoint
train and test splits (cross-validation) and whether this should be
repeated several times. Also, the task defines a target metric for which
a flow should be optimized.

If you want to know more about tasks, try the ["Task tutorial"](../examples/30_extended/tasks_tutorial)

## Running machine learning algorithms and uploading results

In order to upload and share results of running a machine learning
algorithm on a task, we need to create an
[openml.runs.OpenMLRun][]. A run object can be
created by running a [openml.flows.OpenMLFlow][] or a scikit-learn compatible model on a task. We will
focus on the simpler example of running a scikit-learn model.

Flows are descriptions of something runnable which does the machine
learning. A flow contains all information to set up the necessary
machine learning library and its dependencies as well as all possible
parameters.

A run is the outcome of running a flow on a task. It contains all
parameter settings for the flow, a setup string (most likely a command
line call) and all predictions of that run. When a run is uploaded to
the server, the server automatically calculates several metrics which
can be used to compare the performance of different flows to each other.

So far, the OpenML Python connector works only with estimator objects
following the [scikit-learn estimator
API](https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects).
Those can be directly run on a task, and a flow will automatically be
created or downloaded from the server if it already exists.

See ["Simple Flows and Runs"](../examples/20_basic/simple_flows_and_runs_tutorial) for a tutorial covers how to train different machine learning models,
how to run machine learning models on OpenML data and how to share the
results.

## Datasets

OpenML provides a large collection of datasets and the benchmark
[OpenML100](https://docs.openml.org/benchmark/) which consists of a
curated list of datasets.

You can find the dataset that best fits your requirements by making use
of the available metadata. The tutorial ["extended datasets"](../examples/30_extended/datasets_tutorial) which follows explains how to
get a list of datasets, how to filter the list to find the dataset that
suits your requirements and how to download a dataset.

OpenML is about sharing machine learning results and the datasets they
were obtained on. Learn how to share your datasets in the following
tutorial ["Upload"](../examples/30_extended/create_upload_tutorial) tutorial.

# Extending OpenML-Python

OpenML-Python provides an extension interface to connect machine
learning libraries directly to the API and ships a `scikit-learn`
extension. Read more about them in the ["Extensions"](extensions.md) section.

[intro]: examples/20_basic/introduction_tutorial/

