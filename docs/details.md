# Advanced User Guide

This document highlights some of the more advanced features of
`openml-python`. 

## Configuration

The configuration file resides in a directory `~/.config/openml` in the
home directory of the user and is called config (More specifically, it
resides in the [configuration directory specified by the XDG Base
Directory
Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)).
It consists of `key = value` pairs which are separated by newlines. The
following keys are defined:

- apikey: required to access the server.
- server: the server to connect to (default: `https://www.openml.org`).
          For connection to the test server, set this to `https://test.openml.org`.
- cachedir: the root folder where the cache file directories should be created.
    If not given, will default to `~/.openml/cache`
- avoid_duplicate_runs: if set to `True` (default), when certain functions
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
guide](https://docs.openml.org/concepts/).

